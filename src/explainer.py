import torch
from legrad import LeWrapper, LePreprocess
from legrad.utils import visualize
from PIL import Image
from .visual_transformer import ViTWrapper

class LeGradExplainer:
    def __init__(self, clip_model: ViTWrapper):
        self.clip = clip_model
        self.model = LeWrapper(self.clip.model)  # This should work now
        self.model.eval()

        # Override the preprocess with LePreprocess
        self.preprocess = LePreprocess(preprocess=self.clip.preprocess, image_size=self.clip.image_size)

    def compute(self, pil_images, prompt, visualize_result=True):
        if isinstance(pil_images, Image.Image):
            pil_images = [pil_images]

        image_batch = torch.cat([self.preprocess(img).unsqueeze(0) for img in pil_images]).to(self.clip.device)
        text_tokens = self.clip.tokenize(prompt)
        text_embed = self.model.encode_text(text_tokens, normalize=True)

        heatmaps = self.model.compute_legrad_clip(image=image_batch, text_embedding=text_embed)

        if visualize_result:
            for i in range(image_batch.size(0)):
                visualize(heatmaps=heatmaps[i].unsqueeze(0), image=image_batch[i].unsqueeze(0))

        return heatmaps