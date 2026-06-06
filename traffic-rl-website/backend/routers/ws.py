import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from services.fake_trainer import (
    SAFETY_CAP,
    check_convergence,
    complete_training,
    compute_convergence_pct,
    compute_warmup_episodes,
    count_intersections,
    generate_live_payload,
    load_status,
    save_model_file,
)

router = APIRouter(tags=["ws"])


@router.websocket("/ws/live/{session_id}")
async def live_training_stream(websocket: WebSocket, session_id: str) -> None:
    try:
        load_status(session_id)
    except FileNotFoundError:
        await websocket.close(code=4404, reason="Session not found")
        return

    await websocket.accept()

    num_intersections = count_intersections(session_id)
    warmup = compute_warmup_episodes(num_intersections)
    reward_history: list[float] = []
    convergence_streak = 0
    stopped_reason: str | None = None
    final_ep: int | None = None

    last_episode = 0
    last_demand_mode = "auto"
    last_rl_metrics: dict = {}

    try:
        for episode in range(1, SAFETY_CAP + 1):
            convergence_pct = compute_convergence_pct(reward_history)
            payload = generate_live_payload(
                session_id,
                episode,
                convergence_pct=convergence_pct,
                convergence_streak=convergence_streak,
                stopped_reason=stopped_reason,
                final_episode=final_ep,
            )
            await websocket.send_json(payload)

            last_episode = episode
            last_demand_mode = payload.get("demand_mode", "auto")
            last_rl_metrics = dict(payload.get("rl", {}))
            reward_history.append(float(payload["rl"]["reward"]))

            if episode >= warmup:
                convergence_streak, should_stop = check_convergence(reward_history, convergence_streak)
                if should_stop:
                    stopped_reason = "converged"
                    final_ep = episode
                    final_payload = generate_live_payload(
                        session_id,
                        episode,
                        convergence_pct=100,
                        convergence_streak=convergence_streak,
                        stopped_reason=stopped_reason,
                        final_episode=final_ep,
                    )
                    await websocket.send_json(final_payload)
                    break

            await asyncio.sleep(1)

        # Save model file for completed training (converged or safety cap)
        status_data = load_status(session_id)
        save_model_file(
            session_id=session_id,
            final_episode=final_ep or last_episode,
            demand_mode=last_demand_mode,
            final_metrics=last_rl_metrics,
            convergence_pct=100 if stopped_reason == "converged" else compute_convergence_pct(reward_history),
            osm_filename=status_data.get("osm_filename"),
        )
        complete_training(session_id, stopped_reason=stopped_reason, final_episode=final_ep)
    except WebSocketDisconnect:
        return
    finally:
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()
