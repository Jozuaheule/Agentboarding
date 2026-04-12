# Predicates vs. Dynamic Properties

## Short Answer

| | Predicate | Dynamic Property |
|---|---|---|
| **What it is** | A logical structure that can be evaluated as true/false | A predicate whose truth value *changes over time* |
| **Defines** | *What* a state property means | *How* that property transitions from one tick to the next |
| **Form** | `P(args)` — a named condition | `P(args, t+1) ⟺ [rule over P(args, t)]` |
| **When evaluated** | At a single point in time | Across two consecutive time steps |

---

## Predicates — Structures for Evaluating State

A predicate is a logical sentence with arguments that evaluates to **true or false** given the current system state.

```
Occupied(n, t)    — is node n occupied at time t?
HasLuggage(p)     — does passenger p carry luggage?
Edge(n, n')       — is there an edge from n to n' in the graph?
```

Predicates say **what is true** — they describe the state of the world at a given moment. They make no claim about change.

- **Static predicates** are true throughout the whole simulation (e.g., `Edge(n,n')`, `SeatOf(p,n)`).
- **Dynamic predicates** are true at some times and false at others (e.g., `Occupied(n,t)`, `Seated(p,t)`).

---

## Dynamic Properties — Rules That Govern Change

A dynamic property takes a **dynamic predicate** and adds an **update rule** that specifies how its value at time `t+1` is determined by the state at time `t`.

```
Occupied(n, t+1)  ⟺  ∃p : At(p, n, t+1)
At(p, n, t+1)     ⟺  Move(p, n, t)
                  ∨  (At(p, n, t) ∧ Stay(p, t))
```

Dynamic properties say **how the world changes** — they are the executable logic of the model.

---

## Analogy

Think of a spreadsheet:

- A **predicate** is a *column definition* — it names what the cell measures (e.g. "is seat occupied?").
- A **dynamic property** is the *formula in the cell* — it says how the value in row `t+1` is computed from row `t`.

---

## In This Model

| Kind | Examples |
|---|---|
| Static predicates | `Aisle(n)`, `Edge(n,n')`, `SeatOf(p,n)`, `HasLuggage(p)` |
| Dynamic predicates (vocabulary) | `At(p,n,t)`, `Seated(p,t)`, `LuggageState(p,t,s)`, `Intent(p,t,i)` |
| Dynamic properties (update rules) | `At(p,n,t+1) ⟺ Move(p,n,t) ∨ …` |
| Derived/observation predicates | `ObsFree(p,n,t)`, `CloserToSeat(p,n,t)`, `NoProgress(p,t)` — defined from the current state each tick, but with no separate transition rule |

> **Key point:** Not every dynamic predicate has an independent update rule. *Derived* predicates (like `Occupied` or `ObsFree`) are computed directly from other predicates at each tick — they are defined, not transitioned. Only the *core internal states* (`At`, `Seated`, `LuggageState`, etc.) have explicit `t → t+1` transition rules, and those are the true **dynamic properties**.
