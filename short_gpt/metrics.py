

def block_influence(input_hidden_state, output_hidden_state):
    norm_input = input_hidden_state.norm(dim=-1, keepdim=True)
    norm_output = output_hidden_state.norm(dim=-1, keepdim=True)

    delta_mag = (input_hidden_state @ output_hidden_state.T) / (norm_input * norm_output)

    return 1 - delta_mag