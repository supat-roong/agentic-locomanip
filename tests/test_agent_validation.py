from tron2_agent.agent import AgentRunner, ToolCall, validate_call


def test_unknown_tool_rejected():
    assert "unknown tool" in validate_call(ToolCall("fly", {}))


def test_missing_and_extra_args():
    assert "missing" in validate_call(ToolCall("pick", {}))
    assert "unexpected" in validate_call(ToolCall("pick", {"name": "red_cube", "force": 3}))


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
