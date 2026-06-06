# SUMO Simulation Reference Guide

A SUMO simulation has these layers:

| File         | Purpose                                                 |
| ------------ | ------------------------------------------------------- |
| `.sumocfg`   | The launcher / table of contents                        |
| `.net.xml`   | The road network                                        |
| `.rou.xml`   | Traffic demand: vehicles, routes, flows                 |
| `.add.xml`   | Optional extras: detectors, outputs, extra TLS programs |
| Python/TraCI | Live control while SUMO is running                      |

SUMO's network file is basically a directed graph: junctions are nodes, edges are roads, lanes live inside edges, and connections define allowed movements through junctions. SUMO's docs explicitly warn that `.net.xml` is readable but generally not meant to be edited by hand; use NetEdit, netgenerate, netconvert, or plain XML sources when changing structure.

Your current project is a tiny smoke-test network: one four-way crossing, one lane per direction, one signalized controller `A0`, and no real demand scenarios yet. The validation report says it passed structurally with 8 normal edges and exactly one accepted TLS controller, `A0`.

---

## What to Open First

When you get any SUMO network, start with the `.sumocfg`.

Your `cross_smoke.sumocfg` says:

```xml
<input>
    <net-file value="cross_smoke.net.xml"/>
    <route-files value="cross_smoke.empty.rou.xml"/>
</input>

<time>
    <begin value="0"/>
    <end value="3600"/>
</time>
```

So it loads the map, loads the demand file, and simulates one hour.

Officially, a SUMO simulation needs a road network through `--net-file`, and traffic demand normally comes through `--route-files`; route files can contain vehicles, vehicle types, and routes.

---

## Where Everything Lives

| What you want                       | Go to                           | What you change                                   |
| ----------------------------------- | ------------------------------- | ------------------------------------------------- |
| Change simulation duration          | `.sumocfg`                      | `<begin>` / `<end>`                               |
| Change which map is loaded          | `.sumocfg`                      | `<net-file value="..."/>`                         |
| Change which traffic file is loaded | `.sumocfg`                      | `<route-files value="..."/>`                      |
| Add cars                            | `.rou.xml`                      | `<vehicle>` or `<flow>`                           |
| Change car behavior                 | `.rou.xml`                      | `<vType>` acceleration, speed, length, etc.       |
| Add roads/intersections             | NetEdit or generation script    | edges/junctions                                   |
| Change number of lanes              | NetEdit or generation script    | lane count on edges                               |
| Change road speed                   | NetEdit or generation script    | lane/edge speed                                   |
| Change allowed turns                | NetEdit connections mode        | connections between incoming/outgoing edges       |
| Change static traffic-light timing  | `.net.xml` or better `.add.xml` | `<tlLogic>` phases                                |
| Control traffic lights with AI      | Python TraCI                    | `traci.trafficlight.*`                            |
| Read queues, waiting time, vehicles | Python TraCI                    | `traci.lane.*`, `traci.edge.*`, `traci.vehicle.*` |
| Add detectors/sensors               | `.add.xml`                      | induction loops, lane-area detectors              |
| Create outputs/logs                 | `.sumocfg` or `.add.xml`        | tripinfo, edgeData, laneData, TLS output          |

NetEdit is your visual editor. It has network modes for edges, connections, traffic lights, detectors, crossings, etc., and demand modes for routes, vehicles, vehicle types, persons, and stops.

---

## How to Read a `.net.xml`

Search these tags:

```
<edge>
<lane>
<junction>
<connection>
<tlLogic>
<phase>
```

The important distinction:

```xml
<edge id="A0top0" ...>
```

is a normal road.

```xml
<edge id=":A0_0" function="internal">
```

is an internal junction edge. Usually ignore it when writing routes.

### Usable Road Edges

| Edge ID     | Direction       |
| ----------- | --------------- |
| `top0A0`    | top → center    |
| `bottom0A0` | bottom → center |
| `left0A0`   | left → center   |
| `right0A0`  | right → center  |
| `A0top0`    | center → top    |
| `A0bottom0` | center → bottom |
| `A0left0`   | center → left   |
| `A0right0`  | center → right  |

Your file shows exactly these one-lane road edges, plus internal edges used inside the junction.

### Lane IDs

A lane ID is usually `edgeID_laneIndex`. For example:

```
top0A0_0
left0A0_0
A0right0_0
```

---

## How to Read Routes

A `.rou.xml` file describes traffic. SUMO says a vehicle consists of a vehicle type, a route, and the vehicle itself; routes and vehicle types can be reused by many vehicles.

Your current route file is empty:

```xml
<routes>
</routes>
```

So no cars appear because no cars exist.

A valid route is a chain of connected normal edges:

```xml
<route id="south_to_north" edges="bottom0A0 A0top0"/>
```

That means: enter from bottom, pass through `A0`, leave to top.

---

## How to Read Traffic Lights

Search for:

```xml
<tlLogic id="A0">
```

Your current light is:

```xml
<tlLogic id="A0" type="static" programID="0" offset="0">
    <phase duration="42" state="GGggrrrrGGggrrrr"/>
    <phase duration="3"  state="yyyyrrrryyyyrrrr"/>
    <phase duration="42" state="rrrrGGggrrrrGGgg"/>
    <phase duration="3"  state="rrrryyyyrrrryyyy"/>
</tlLogic>
```

This is fixed-time control: phase 0, phase 1, phase 2, phase 3, repeat.

The annoying part: the state string does not directly mean "north road is green." Each character controls a **link**, meaning one movement from an incoming lane to an outgoing lane. SUMO says the mapping comes from the `linkIndex` attribute in `<connection>` elements, and each character in a phase state corresponds to one signal/link.

Example from your network:

```xml
<connection from="bottom0A0" to="A0top0" tl="A0" linkIndex="9" dir="s"/>
```

That means character index `9` in the traffic-light state controls the bottom-to-top straight movement.

### Signal Letter Meanings

| Letter | Meaning             |
| ------ | ------------------- |
| `r`    | Red                 |
| `y`    | Yellow              |
| `g`    | Green but yields    |
| `G`    | Green with priority |

---

## What Not to Touch Blindly

Do **not** randomly edit these unless you know why:

```xml
<request>
<foes>
<response>
<junction shape="...">
<edge id=":A0_...">
```

Those are generated conflict/right-of-way/internal geometry details. Messing with them by hand is how networks become cursed.

---

## The Proper Way to Change Things

For simple demand, edit `.rou.xml`.

For roads, lanes, junctions, turns, and traffic lights, use:

```bash
netedit cross_smoke.net.xml
```

For generated grid networks, change the generation command instead of patching the final `.net.xml`. Your manifest shows this network came from `netgenerate` with a 1×1 grid, 200m arms, one lane, speed 13.89, and `A0` set as the only traffic light.
