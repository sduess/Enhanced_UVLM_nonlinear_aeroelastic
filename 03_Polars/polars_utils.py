def get_case_name(u_inf, alpha_deg, use_polars, lifting_only):
    """
    Generate a descriptive case name based on simulation parameters.

    Parameters:
    - u_inf (float): Cruise flight speed.
    - alpha_deg (float): Angle of attack in degrees.
    - use_polars (bool): Whether polar corrections are applied.
    - lifting_only (bool): Whether only lifting bodies are considered.

    Returns:
    - case_name (str): Descriptive case name.
    """
    alpha_str = int(alpha_deg * 100)
    if alpha_deg < 0:
        alpha_str = 'm' + str(abs(alpha_str))
    case_name = 'superflexop_uinf{}_alpha{}_p{}_f{}'.format(int(u_inf),
                                                            alpha_str,
                                                            int(use_polars),
                                                            int(not lifting_only))
    return case_name
