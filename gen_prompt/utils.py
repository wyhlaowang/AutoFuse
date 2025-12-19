import os
import re
import shutil
import torch


def get_first_sentence(text, ending_punctuation = ',.'):
    for index, char in enumerate(text):
        if char in ending_punctuation:
            return text[:index + 1]
    return text


def copy_files(src_dir, dest_dir, file_extension):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            src_file_path = os.path.join(root, file)
            # check file type
            if src_file_path.endswith(file_extension):
                dest_file_path = os.path.join(dest_dir, file)
                # copy
                shutil.copy(src_file_path, dest_file_path)
                print(f"copying {src_file_path} to {dest_file_path}")


def get_tv(inputs):
    diff1 = inputs[:, :, :, :-1] - inputs[:, :, :, 1:]
    diff2 = inputs[:, :, :-1, :] - inputs[:, :, 1:, :]
    diff3 = inputs[:, :, 1:, :-1] - inputs[:, :, :-1, 1:]
    diff4 = inputs[:, :, :-1, :-1] - inputs[:, :, 1:, 1:]

    tv = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    
    return tv

def entropy_t4d(images, max=1, if_norm=False):
    if if_norm:
        mean = images.mean(dim=[-1,-2,-3], keepdim=True)
        std = images.std(dim=[-1,-2,-3], keepdim=True)
        images = (images - mean) / std

    B, C, _, _ = images.shape
    entropies = torch.zeros(B)
    for i in range(B):
        image = images[i]
        image_flat = image.view(C, -1)
        hist = torch.histc(image_flat, bins=256, min=0, max=max)
        prob = hist / torch.sum(hist)
        prob = prob[prob > 0]
        entropies[i] = -torch.sum(prob * torch.log2(prob))
    
    return entropies


def remove_before_be(text):
    # Use regex to find the first occurrence of ' is ' or ' be ' and remove everything before it
    result = re.sub(r'^.*?\b(is|be)\b', '\1', text, count=1)
    return result


def extract_phrases(text):
    pattern = r'\d+\.\s+"?([^"\n]+)"?|\*\s+"?([^"\n]+)"?|"([^"]+)"'
    matches = re.findall(pattern, text)
    results = [re.split(r',', match[0] or match[1] or match[2])[0].strip().lower() for match in matches]

    return results


def add_prefix_to_words(input_string, prefix="image"):
    words = [word.strip() for word in input_string.split(",")]  
    modified_words = [f"{prefix} {word}" for word in words]  
    return ", ".join(modified_words)  


def if_text_in(text, target_string="no visible degradation"):
    return target_string.lower() in text.lower()


def text_to_dict_llama1(text):
    matches = re.findall(r'\d+\.\s+(.*?)[-:>\u2013]+\s+(.*)', text)
    result_dict = {key.strip().lower(): value.strip().lower() for key, value in matches}
    return result_dict


def text_to_dict_llama3(text):
    matches = re.findall(r'^(?:[-]|\d+\.)\s*([a-z\s]+?)\s*[:]\s*([a-z\s]+)$', text.lower(), re.MULTILINE)
    result_dict = {key.strip(): value.strip() for key, value in matches}
    return result_dict


def tensor2PIL(im):
    im = im.repeat(3, 1, 1) if im.shape[0] == 1 else im
    np_array = im.cpu().numpy().astype(np.uint8)
    PIL_image = Image.fromarray(rearrange(np_array, "c h w -> h w c"))
    return PIL_image


def to_y(im):
    im_ra = rearrange(im.cpu(), 'c h w -> h w c').numpy()
    im_ra = np.repeat(im_ra, 3, axis=-1) if im_ra.shape[-1] == 1 else im_ra
    im_ycrcb = cv2.cvtColor(im_ra, cv2.COLOR_RGB2YCrCb)
    im_y = torch.from_numpy(im_ycrcb[:, :, 0]).unsqueeze(0)

    return im_y


def extract_text_pairs(text):
    text = text.lower()
    text = re.sub(r'[\*\->\-]', '', text)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    pairs = {}
    pattern = r"^(.*?):\s*(.*?)$"
    for line in lines:
        match = re.match(pattern, line)
        if match:
            key, value = match.groups()
            pairs[key.strip()] = value.strip()
    
    return pairs