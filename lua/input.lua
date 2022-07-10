
local input = {}

do
	local raw = {
		'escape', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', -- 0x09
		'num9', 'num0', 'minus', 'equal', 'backspace', -- 0x0e
		'tab', -- 0x0f
		'q', 'w', 'e', 'r', -- 0x13
		't', 'y', 'u', 'i', 'o', 'p', 'left_bracket', 'right_bracket', 'enter', -- 0x1c
		'lcontrol', -- 0x1d
		'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'semicolon', -- 0x27
		'quote', 'backtick', 'lshift', -- 0x2a
		'backslash', 'z', 'x', 'c', 'v', 'b', 'n', -- 0x31
		'm', 'comma', 'dot', 'slash', 'rshift', -- 0x36
		'asterisk', -- 0x37
		'lalt', -- 0x38
		'space', -- 0x39
		'caps_lock', -- 0x3a
		'f1', -- 0x3b
		'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', -- 0x43
		'f10', -- 0x44
		'pause', -- 0x45
		'scroll_lock', -- 0x46
		'kp7', -- 0x47
		'kp8', -- 0x48
		'kp9', -- 0x49
		'kp_minus', -- 0x4a
		'kp4', -- 0x4b
		'kp5', -- 0x4c
		'kp6', -- 0x4d
		'kp_plus', -- 0x4e
		'kp1', -- 0x4f
		'kp2', -- 0x50
		'kp3', -- 0x51
		'kp0', -- 0x52
		'kp_dot', -- 0x53
		nil, nil, nil, -- 0x56
		'f11', -- 0x57
		'f12', -- 0x58
		[0xe01c] = 'kp_enter',
		[0xe01d] = 'rcontrol',
		[0xe035] = 'kp_slash',
		[0xe037] = 'print_screen',
		[0xe038] = 'ralt',
		[0xe045] = 'num_lock',
		[0xe047] = 'home',
		[0xe048] = 'up',
		[0xe049] = 'page_up',
		[0xe04b] = 'left',
		[0xe04d] = 'right',
		[0xe04f] = 'end',
		[0xe050] = 'down',
		[0xe051] = 'page_down',
		[0xe052] = 'insert',
		[0xe053] = 'delete',
		[0xe05b] = 'lmeta',
		[0xe05c] = 'rmeta',
	}

	local scancodes = {}
	for k, v in pairs(raw) do
		scancodes[k] = v
		scancodes[v] = k
	end

	input.scancodes = scancodes
end

input.mouse_buttons = {}
for i = 0, 20 do
	input.mouse_buttons[i] = 'm'..(i+1)
end
for i = 0, #input.mouse_buttons do
	input.mouse_buttons[input.mouse_buttons[i]] = i
end

input.is_down = {}

return input
