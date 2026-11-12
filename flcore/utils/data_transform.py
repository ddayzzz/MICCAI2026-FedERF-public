from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop, InterpolationMode


def get_clip_model_image_transform(image_size=224):
    return Compose([
        Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(image_size),
        # _convert_image_to_rgb,  # 已经默认RGB
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])