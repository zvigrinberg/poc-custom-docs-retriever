from re import Match


def get_current_block(code: str, match: Match):
    curly_brackets_counter = 1
    internal_offset = 0
    current_block_body = code[match.end():]
    while curly_brackets_counter > 0 or internal_offset == 0:
        left_bracket_ind = current_block_body[internal_offset:].find("{")
        right_bracket_ind = current_block_body[internal_offset:].find("}")
        if left_bracket_ind < right_bracket_ind and left_bracket_ind != -1:
            curly_brackets_counter += 1
            internal_offset = internal_offset + left_bracket_ind + 1
        else:
            curly_brackets_counter -= 1
            internal_offset = internal_offset + right_bracket_ind + 1
    current_block = code[match.start(): match.end() + 1 + internal_offset]
    return current_block
