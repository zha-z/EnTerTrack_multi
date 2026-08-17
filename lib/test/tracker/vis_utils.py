import numpy as np
try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
from PIL import Image, ImageDraw, ImageFont

############## used for visulize eliminated tokens #################
def get_keep_indices(decisions):
    keep_indices = []
    for i in range(3):
        if i == 0:
            keep_indices.append(decisions[i])
        else:
            keep_indices.append(keep_indices[-1][decisions[i]])
    return keep_indices


def gen_masked_tokens(tokens, indices, alpha=0.2):
    # indices = [i for i in range(196) if i not in indices]
    indices = indices[0].astype(int)
    tokens = tokens.copy()
    tokens[indices] = alpha * tokens[indices] + (1 - alpha) * 255
    return tokens


def recover_image(tokens, H, W, Hp, Wp, patch_size):
    # image: (C, 196, 16, 16)
    image = tokens.reshape(Hp, Wp, patch_size, patch_size, 3).swapaxes(1, 2).reshape(H, W, 3)
    return image


def pad_img(img):
    height, width, channels = img.shape
    im_bg = np.ones((height, width + 8, channels)) * 255
    im_bg[0:height, 0:width, :] = img
    return im_bg


def draw_text(img, text, xy):
    if cv2 is not None:
        cv2.putText(
            img,
            text,
            xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )
        return img

    pil_img = Image.fromarray(np.asarray(img, dtype=np.uint8))
    draw = ImageDraw.Draw(pil_img)
    draw.text(xy, text, fill=(0, 0, 0), font=ImageFont.load_default())
    return np.asarray(pil_img)


def gen_visualization(image, mask_indices, patch_size=16):
    # image [224, 224, 3]
    # mask_indices, list of masked token indices

    # mask mask_indices need to cat
    # mask_indices = mask_indices[::-1]
    num_stages = len(mask_indices)
    #for i in range(1, num_stages):
    #    mask_indices[i] = np.concatenate([mask_indices[i-1], mask_indices[i]], axis=1)

    # keep_indices = get_keep_indices(decisions)
    image = np.asarray(image)
    H, W, C = image.shape
    Hp, Wp = H // patch_size, W // patch_size

    total_tokens = Hp * Wp
    image_tokens = image.reshape(Hp, patch_size, Wp, patch_size, 3).swapaxes(1, 2).reshape(Hp * Wp, patch_size, patch_size, 3)


    stages = []
    keep_ratios = []
    for i in range(num_stages):
        masked_img = recover_image(gen_masked_tokens(image_tokens, mask_indices[i]), H, W, Hp, Wp, patch_size)
        keep_num = total_tokens - len(mask_indices[i][0])
        keep_ratio = keep_num / total_tokens
        keep_ratios.append(keep_ratio)

        masked_img = draw_text(masked_img, f"keep: {keep_num} ({keep_ratio:.2%})", (10, 20))
        stages.append(masked_img)
    imgs = [image] + stages
    imgs = [pad_img(img) for img in imgs]
    viz = np.concatenate(imgs, axis=1)
    return viz
