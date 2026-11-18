from tron2_agent.agent import AgentRunner, ToolCall, validate_call


def test_unknown_tool_rejected():
    assert "unknown tool" in validate_call(ToolCall("fly", {}))


def test_missing_and_extra_args():
    assert "missing" in validate_call(ToolCall("move_eef", {"x": 1, "y": 0, "z": 0.7}))
    assert "unexpected" in validate_call(ToolCall("gripper", {"action": "open", "force": 3}))
    assert "one of" in validate_call(ToolCall("gripper", {"action": "squeeze"}))


def test_string_number_coerced():
    c = ToolCall("navigate_to", {"x": "1.5", "y": 0})
    assert validate_call(c) is None and c.args["x"] == 1.5


def test_enum_and_limits():
    assert "one of" in validate_call(ToolCall("look", {"camera": "tail"}))
    assert validate_call(ToolCall("rotate", {"angle_deg": 720})) is not None
    assert validate_call(ToolCall("navigate_to", {"x": 50, "y": 0})) is not None


def test_bool_coercion_for_done():
    c = ToolCall("done", {"success": "true", "summary": "ok"})
    assert validate_call(c) is None and c.args["success"] is True


def test_old_images_dropped():
    msgs = [{"role": "user", "content": "a", "images": [b"1"]}, {"role": "user", "content": "b", "images": [b"2"]}]
    AgentRunner._drop_old_images(msgs, keep=1)
    assert "images" not in msgs[0] and msgs[1]["images"] == [b"2"]


def test_with_reason_adds_required_reason_to_every_tool():
    from tron2_agent.agent import TOOLS, with_reason

    tools = with_reason(TOOLS)
    for t in tools:
        params = t["function"]["parameters"]
        assert "reason" in params["properties"] and "reason" in params["required"]
    assert "reason" not in TOOLS[0]["function"]["parameters"]["properties"]  # original untouched


def test_demo_call_str():
    from tron2_agent.demo import _call_str

    assert _call_str("gripper", {"action": "close"}) == 'gripper(action="close")'
    assert _call_str("rotate", {"angle_deg": 45.0}) == "rotate(angle_deg=45)"
    assert _call_str("done", {"success": True, "summary": "x"}) == "done(success=true)"
