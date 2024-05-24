import folder_paths


def vae_list():
    # Borrowed verbatim from comfyui's implementations.
    vaes = folder_paths.get_filename_list("vae")
    approx_vaes = folder_paths.get_filename_list("vae_approx")
    sdxl_taesd_enc = False
    sdxl_taesd_dec = False
    sd1_taesd_enc = False
    sd1_taesd_dec = False

    for v in approx_vaes:
        if v.startswith("taesd_decoder."):
            sd1_taesd_dec = True
        elif v.startswith("taesd_encoder."):
            sd1_taesd_enc = True
        elif v.startswith("taesdxl_decoder."):
            sdxl_taesd_dec = True
        elif v.startswith("taesdxl_encoder."):
            sdxl_taesd_enc = True
    if sd1_taesd_dec and sd1_taesd_enc:
        vaes.append("taesd")
    if sdxl_taesd_dec and sdxl_taesd_enc:
        vaes.append("taesdxl")
    return vaes


def load_taesd(name):
    # Borrowed verbatim from comfyui's implementations
    sd = {}
    approx_vaes = folder_paths.get_filename_list("vae_approx")

    encoder = next(filter(lambda a: a.startswith("{}_encoder.".format(name)), approx_vaes))
    decoder = next(filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes))

    enc = comfy.utils.load_torch_file(folder_paths.get_full_path("vae_approx", encoder))
    for k in enc:
        sd["taesd_encoder.{}".format(k)] = enc[k]

    dec = comfy.utils.load_torch_file(folder_paths.get_full_path("vae_approx", decoder))
    for k in dec:
        sd["taesd_decoder.{}".format(k)] = dec[k]

    if name == "taesd":
        sd["vae_scale"] = torch.tensor(0.18215)
    elif name == "taesdxl":
        sd["vae_scale"] = torch.tensor(0.13025)
    return sd