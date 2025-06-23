import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPReIDImageOnlyWrapper(nn.Module):
    """Simplified image-only wrapper that completely avoids conditional logic"""
    
    def __init__(self, original_model):
        super(CLIPReIDImageOnlyWrapper, self).__init__()
        
        # Copy essential components
        self.image_encoder = original_model.image_encoder
        self.bottleneck = original_model.bottleneck
        self.bottleneck_proj = original_model.bottleneck_proj
        self.model_name = original_model.model_name
        self.neck_feat = original_model.neck_feat
        
        # Set to eval mode
        self.eval()
        
    def forward(self, x):
        """Simplified forward pass for image inference only"""
        # Process based on model architecture
        if self.model_name == 'ViT-B-16':
            # ViT-B-16 path without camera/view embeddings
            image_features_last, image_features, image_features_proj = self.image_encoder(x, None) 
            img_feature_last = image_features_last[:,0]
            img_feature = image_features[:,0]
            img_feature_proj = image_features_proj[:,0]
        else:  # RN50
            image_features_last, image_features, image_features_proj = self.image_encoder(x) 
            img_feature_last = F.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = F.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]

        # Apply bottleneck layers
        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 
        
        # Always return concatenated features (avoid string comparison)
        # This assumes 'after' mode which is most common for inference
        return torch.cat([feat, feat_proj], dim=1)

class CLIPReIDMinimalWrapper(nn.Module):
    """Minimal wrapper that extracts and simplifies the core functionality"""
    
    def __init__(self, original_model):
        super(CLIPReIDMinimalWrapper, self).__init__()
        
        # Extract the core vision transformer or CNN backbone
        self.backbone = self._extract_backbone(original_model)
        self.bottleneck = original_model.bottleneck
        self.bottleneck_proj = original_model.bottleneck_proj
        self.model_name = original_model.model_name
        
        # Set to eval mode
        self.eval()
        
    def _extract_backbone(self, original_model):
        """Extract the core backbone without complex CLIP operations"""
        if hasattr(original_model.image_encoder, 'visual'):
            return original_model.image_encoder.visual
        else:
            return original_model.image_encoder
    
    def forward(self, x):
        """Forward pass using simplified backbone"""
        # Use the backbone directly
        features = self.backbone(x)
        
        # Handle different output formats
        if isinstance(features, tuple):
            # If multiple outputs, use the main feature
            if len(features) >= 2:
                img_feature = features[1]  # Middle features
                img_feature_proj = features[2] if len(features) > 2 else features[1]  # Projected features
            else:
                img_feature = img_feature_proj = features[0]
        else:
            # Single output
            img_feature = img_feature_proj = features
            
        # Handle tensor shapes based on model type
        if self.model_name == 'ViT-B-16':
            if len(img_feature.shape) == 3:  # [batch, seq_len, dim]
                img_feature = img_feature[:, 0]  # Take CLS token
            if len(img_feature_proj.shape) == 3:
                img_feature_proj = img_feature_proj[:, 0]
        elif self.model_name == 'RN50':
            if len(img_feature.shape) == 4:  # [batch, channels, height, width]
                img_feature = F.avg_pool2d(img_feature, img_feature.shape[2:4]).view(x.shape[0], -1)
            if len(img_feature_proj.shape) == 4:
                img_feature_proj = F.avg_pool2d(img_feature_proj, img_feature_proj.shape[2:4]).view(x.shape[0], -1)
        
        # Apply bottleneck layers
        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 
        
        # Return concatenated features
        return torch.cat([feat, feat_proj], dim=1)