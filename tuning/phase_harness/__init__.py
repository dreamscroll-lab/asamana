"""Per-step cognition stages run dry with tracing, one module per stage, each exposing ``run_<stage>``.

``common`` restores a world and sandboxes an agent's writes; ``scenario`` turns a scenario's setup into
world and agent state. Stage modules import only from those two, never from each other.
"""
