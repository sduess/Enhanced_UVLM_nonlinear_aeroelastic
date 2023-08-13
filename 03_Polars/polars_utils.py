
def get_case_name(u_inf, alpha_deg, use_polars, lifting_only):
    alpha_str = int(alpha_deg*100)
    if alpha_deg < 0:
        alpha_str = 'm' + str(abs(alpha_str))
    case_name = 'superflexop_uinf{}_alpha{}_p{}_f{}'.format(int(u_inf),
                                                            alpha_str,
                                                            int(use_polars),
                                                            int(not lifting_only))
    return case_name