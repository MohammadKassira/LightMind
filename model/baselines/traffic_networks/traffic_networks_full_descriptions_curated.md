# Traffic Networks Inventory and Descriptions

This file lists the curated traffic networks, a short description of each one, and the number of **traffic-light intersections**.

## Tiny debugging and sanity-check networks

| Network | Traffic-light intersections | Description | Best use |
|---|---:|---|---|
| `cross_smoke` | 1 | Very small local smoke-test network with a single signalized intersection. | Fast debugging |
| `cologne1` | 1 | Tiny RESCO Cologne scenario with one traffic light. | Tiny real-network sanity check |
| `ingolstadt1` | 1 | Tiny RESCO Ingolstadt scenario with one traffic light. | Tiny real-network sanity check |
| `grid6_smoke` | 6 | Small local multi-intersection grid used for early debugging and quick checks. | Fast debugging |

## Main practical training networks

| Network | Traffic-light intersections | Description | Best use |
|---|---:|---|---|
| `cologne3` | 3 | Small real-city RESCO Cologne network. It is realistic and quick to train, so it works well as a stepping-stone network. | Small real-network training / sanity benchmark |
| `ingolstadt7` | 7 | Medium real-city RESCO Ingolstadt network with more coordination than the tiny maps. | Core training network |
| `cologne8_resco` | 8 | Medium RESCO Cologne network already integrated into the project. | Core training network |
| `most` | 12 | Monaco SUMO Traffic scenario with a different city structure and traffic pattern from the RESCO maps. | Strong practical training candidate |
| `bologna_pasubio` | 15 | Realistic urban Bologna scenario from DLR SUMO scenarios. | Strong practical training candidate |
| `arterial4x4` | 16 | Structured arterial RESCO network. Good for corridor-style coordination experiments. | Core structured training network |
| `ingolstadt21` | 24 | Larger and more complex real-city Ingolstadt RESCO network. | One of the strongest main training networks |

## Recommended final training story

The stronger final training story should rely more on:
- `ingolstadt7`
- `cologne8_resco`
- `most`
- `bologna_pasubio`
- `arterial4x4`
- `ingolstadt21`

