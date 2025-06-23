import os
import time
from config import cfg
import argparse
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from processor.processor_clipreid_stage2 import do_inference
from utils.logger import setup_logger
import torchvision.transforms as T
import torch
from PIL import Image
import numpy as np

from torchinfo import summary
import torch


def collect_image_paths(val_dir):
    image_paths = []
    for person_dir in os.listdir(val_dir):
        person_path = os.path.join(val_dir, person_dir)
        if os.path.isdir(person_path):
            for img in os.listdir(person_path):
                if img.endswith('.jpg') or img.endswith('.png'):
                    image_paths.append(os.path.join(person_path, img))

    person_ids = [int(dir_name) for dir_name in os.listdir(val_dir) if dir_name.isdigit()]
    num_person_ids = len(person_ids)
    logger.info("Number of person IDs in validation set: {}".format(num_person_ids))
    n_images = len(image_paths)
    logger.info("Number of images in validation set: {}".format(n_images))
    return image_paths, person_ids

def embedd_images(model, image_paths, val_transforms):
    n_images = len(image_paths)
    embeddings = []
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for img_path in image_paths:
            img = Image.open(img_path).convert('RGB')
            img = val_transforms(img).unsqueeze(0).to("cuda")
            feat = model(img, get_image=True)
            embeddings.append(feat.squeeze().cpu().numpy())
    end_time = time.time()
    print("Time taken to extract {} embeddings: {:.2f} seconds = {:3} FPS".format(n_images, end_time - start_time, n_images / (end_time - start_time)))
    return embeddings

def rank_by_similarity(true_query_person_id, query_embedding, gallery_image_paths, gallery_embeddings):
    # Compute cosine similarity between query_embedding and gallery_embeddings
    similarities = np.dot(gallery_embeddings, query_embedding)
    # Sort indices by similarity in descending order
    ranks = np.argsort(similarities)[::-1]
    
    # Get the person IDs of the gallery images
    gallery_person_ids = [os.path.basename(os.path.dirname(path)) for path in gallery_image_paths]

    gallery_person_ids = np.array(gallery_person_ids)
    are_identical = (gallery_person_ids[ranks] == true_query_person_id).astype(int)
    # index of the true query person ID in the gallery
    true_query_index = list(are_identical).index(1) if 1 in are_identical else len(gallery_embeddings)
    return similarities, ranks, true_query_index, are_identical


def eval_with_clustering(model, val_dir, val_transforms):
    image_paths, person_ids = collect_image_paths(val_dir)
    num_person_ids = len(person_ids)
    embeddings = embedd_images(model, image_paths, val_transforms)

    # cluster all embeddings with k-means
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_person_ids, random_state=42)
    kmeans.fit(embeddings)

    labels = kmeans.labels_
    ground_truth = [int(os.path.basename(os.path.dirname(path))) for path in image_paths]

    # calculate NMI, ARI
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    nmi = normalized_mutual_info_score(ground_truth, labels)
    ari = adjusted_rand_score(ground_truth, labels)
    logger.info("NMI: {:.4f}, ARI: {:.4f}".format(nmi, ari))
    print("NMI: {:.4f}, ARI: {:.4f}".format(nmi, ari))

    # compile collage per cluster
    output_dir = '/home/oron_nir/clipreid/collages'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    compile_collage_per_cluster(image_paths, labels, output_dir)


def average_precision(are_identical):
    total_relevant = np.sum(are_identical)
    if total_relevant == 0:
        return 0.0
    # Calculate average precision
    ks = np.arange(1, len(are_identical) + 1)
    cumsum_ids = np.cumsum(are_identical)
    precision_at_k = cumsum_ids / ks
    average_precision = np.sum(precision_at_k * are_identical) / total_relevant
    return average_precision


def eval_with_ranking_cmc(model, gallery_repo, query_repo):
    gallery_image_paths, person_ids = collect_image_paths(gallery_repo)
    gallery_embeddings = embedd_images(model, gallery_image_paths, val_transforms)
    gallery_image_names = set(os.path.basename(path) for path in gallery_image_paths)
    query_image_paths, _ = collect_image_paths(query_repo)
    query_embeddings = embedd_images(model, query_image_paths, val_transforms)

    start_time = time.time()
    n_skipped = 0
    rank_1s, rank_5s, rank_10s, APs = [], [], [], []
    for query_path, queryimage_embed in zip(query_image_paths, query_embeddings):
        true_query_person_id = os.path.basename(os.path.dirname(query_path))
        # Compute cosine similarity between query_embedding and gallery_embeddings
        similarities, ranks, true_query_index, are_identical = rank_by_similarity(true_query_person_id, queryimage_embed, gallery_image_paths, gallery_embeddings)
        # Compute ranks
        rank_1s.append(1 if true_query_index < 1 else 0)
        rank_5s.append(1 if true_query_index < 5 else 0)
        rank_10s.append(1 if true_query_index < 10 else 0)
        APs.append(average_precision(are_identical))
    rank_1 = np.mean(rank_1s)
    rank_5 = np.mean(rank_5s)
    rank_10 = np.mean(rank_10s)
    mAP = np.mean(APs)
    n_query = len(rank_1s)
    end_time = time.time()
    total_time = end_time - start_time
    print("Total inference time: {:.2f} seconds".format(total_time))
    print("Rank-1: {:.3%}, Rank-5: {:.3%}, Rank-10: {:.3%}, mAP: {:.3%}, n_query: {}".format(rank_1, rank_5, rank_10, mAP, n_query))
    print("Skipped {} queries that were already in the gallery.".format(n_skipped))
    return rank_1, rank_5, rank_10, mAP, n_query


def compile_collage_per_cluster(image_paths, labels, output_dir):
    """Compiles a collage of images for each cluster and saves it to the output directory."""
    from collections import defaultdict
    import matplotlib.pyplot as plt
    import shutil

    # clear the output directory if it exists
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    clusters = defaultdict(list)
    for img_path, label in zip(image_paths, labels):
        clusters[label].append(img_path)

    for label, paths in clusters.items():
        if not paths:
            continue
        images = [Image.open(path) for path in paths]
        collage_width = 5 * 256  # Assuming each image is 256x256
        collage_height = (len(images) // 5 + 1) * 256
        collage = Image.new('RGB', (collage_width, collage_height))

        for i, img in enumerate(images):
            x = (i % 5) * 256
            y = (i // 5) * 256
            collage.paste(img.resize((256, 256)), (x, y))

        collage.save(os.path.join(output_dir, f'cluster_{label}.png'))

def export_to_onnx(model, onnx_file_path, input_size=(1, 3, 256, 128), opset_version=11):
    """
    Export the model to ONNX format.
    """
    
    # Create ONNX-compatible wrapper that eliminates conditional logic
    print(f"Creating ONNX wrapper for {model.model_name} model...")

    # Try different wrapper approaches
    try:
        # First try the minimal wrapper
        from model.onnx_wrapper import CLIPReIDMinimalWrapper
        onnx_model = CLIPReIDMinimalWrapper(model)
        wrapper_type = "minimal"
    except Exception as e:
        print(f"Minimal wrapper failed: {e}")
        # Fallback to image-only wrapper
        from model.onnx_wrapper import CLIPReIDImageOnlyWrapper
        onnx_model = CLIPReIDImageOnlyWrapper(model)
        wrapper_type = "image-only"

    onnx_model.eval()
    onnx_model.to("cuda")

    # Test the wrapper with dummy input first
    dummy_input = torch.randn(1, 3, 256, 128).to("cuda")
    print(f"Testing {wrapper_type} ONNX wrapper...")
    try:
        with torch.no_grad():
            wrapper_output = onnx_model(dummy_input)
            print(f"Wrapper output shape: {wrapper_output.shape}")
    except Exception as e:
        print(f"Wrapper test failed: {e}")
        # If wrapper fails, we can't proceed with ONNX export
        print("Skipping ONNX export due to wrapper failure")
    else:
        print(f"Exporting model to ONNX format at {onnx_file_path}...")
        
        # Try different opset versions for compatibility
        opset_versions = [13, 14, 15, 11]  # Start with newer versions first
        
        for opset_version in opset_versions:
            try:
                print(f"Attempting ONNX export with opset version {opset_version}...")
                
                torch.onnx.export(
                    onnx_model, 
                    dummy_input, 
                    onnx_file_path, 
                    export_params=True, 
                    opset_version=opset_version,
                    do_constant_folding=True, 
                    input_names=['input'], 
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    },
                    verbose=False
                )
                
                print(f"Successfully exported ONNX model with opset version {opset_version}")
                break
                
            except Exception as e:
                print(f"ONNX export failed with opset version {opset_version}: {e}")
                if opset_version == opset_versions[-1]:  # Last attempt
                    print("All ONNX export attempts failed")
                    continue
                else:
                    continue
        else:
            print("ONNX export failed with all opset versions")

        # Only validate if export was successful
        if os.path.exists(onnx_file_path):
            # Validate the ONNX model
            print("Validating ONNX model...")
            import onnx
            try:
                onnx_model_check = onnx.load(onnx_file_path)
                onnx.checker.check_model(onnx_model_check)
                print("ONNX model validation passed")
                
                # Print model info
                print(f"ONNX model inputs: {[input.name for input in onnx_model_check.graph.input]}")
                print(f"ONNX model outputs: {[output.name for output in onnx_model_check.graph.output]}")
                
            except Exception as e:
                print(f"ONNX model validation failed: {e}")

            # Test ONNX model inference
            print("Testing ONNX model inference...")
            try:
                import onnxruntime as ort
                
                # Create ONNX Runtime session
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                ort_session = ort.InferenceSession(onnx_file_path, providers=providers)
                
                # Test inference
                dummy_input_np = dummy_input.cpu().numpy()
                ort_inputs = {ort_session.get_inputs()[0].name: dummy_input_np}
                ort_outputs = ort_session.run(None, ort_inputs)
                
                print(f"ONNX Runtime output shape: {ort_outputs[0].shape}")
                
                # Compare outputs
                pytorch_output = wrapper_output.cpu().numpy()
                onnx_output = ort_outputs[0]
                max_diff = np.max(np.abs(pytorch_output - onnx_output))
                print(f"Max difference between PyTorch and ONNX outputs: {max_diff}")
                
                if max_diff < 1e-4:
                    print("✓ ONNX model outputs match PyTorch model outputs")
                else:
                    print("⚠ ONNX model outputs differ from PyTorch model outputs")
                
            except ImportError:
                print("ONNX Runtime not installed, skipping inference test")
                print("To install: pip install onnxruntime-gpu")
            except Exception as e:
                print(f"ONNX model inference test failed: {e}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_clipreid.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    # --config_file configs/person/vit_clipreid.yml TEST.WEIGHT '/home/oron_nir/clipreid/models/Market1501_clipreid_ViT-B-16_60.pth' DATASETS.NAMES 'market1501'
    # args.config_file = 'configs/person/vit_clipreid.yml'
    # args.TEST_WEIGHT = '/home/oron_nir/clipreid/models/Market1501_clipreid_ViT-B-16_60.pth'
    # args.DATASETS_NAMES = 'market1501'

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param(cfg.TEST.WEIGHT)
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    # Get model summary with FLOPs and memory
    summary_info = summary(model, input_size=(1, 3, 256, 128), verbose=0)

    print(f"Model FLOPs: {summary_info.total_mult_adds / 1e9:.2f} GFLOPs")
    print(f"Model parameters: {summary_info.total_params / 1e6:.2f} M")
    print(f"Model size: {summary_info.total_param_bytes / 1e6:.2f} MB")

    # fix start

    # Replace the ONNX export section with this improved version:

    # serialize the model to a ONNX file
    print("Exporting model to ONNX format...")
    model.eval()
    models_dir = '/home/oron_nir/clipreid/models/'
    os.makedirs(models_dir, exist_ok=True)
    onnx_file_path = os.path.join(models_dir, 'Market1501_clipreid_ViT-B-16_60.onnx')

    export_to_onnx(model, onnx_file_path, input_size=(1, 3, 256, 128), opset_version=11)

    # fix end




    # serialize the model to a ONNX file
    model.eval()
    models_dir = '/home/oron_nir/clipreid/models/'
    onnx_file_path = os.path.join(models_dir, 'Market1501_clipreid_ViT-B-16_60.onnx')
    dummy_input = torch.randn(1, 3, 256, 128).to("cuda")
    print(f"\n\nExporting model to ONNX format at {onnx_file_path}...")
    torch.onnx.export(model, dummy_input, onnx_file_path, export_params=True, opset_version=11,
                      do_constant_folding=True, input_names=['input'], output_names=['output'])
    logger.info(f"ONNX model exported to {onnx_file_path}")

    # Check if ONNX file is valid
    import onnx

    # Load the ONNX model
    onnx_model = onnx.load("model.onnx")

    # Check the model
    onnx.checker.check_model(onnx_model)

    
    # Evaluate with clustering on the validation set
    datasets_paths = [
        '/home/oron_nir/clipreid/data/PRCC/prcc/rgb/test/A',
        # '/home/oron_nir/clipreid/data/PRCC/prcc/rgb/test/B',
        # '/home/oron_nir/clipreid/data/PRCC/prcc/rgb/test/C',
        # '/home/oron_nir/clipreid/data/PRCC/prcc/rgb/val'
    ]
    for dataset_path in datasets_paths:
        if not os.path.exists(dataset_path):
            print(f"Dataset path {dataset_path} does not exist. Skipping evaluation.")
            continue
        dataset_name = os.path.basename(dataset_path)
        print(f"Evaluating dataset: {dataset_name}")
        print(f"Starting clustering evaluation on {dataset_name}...")
        eval_with_clustering(model, dataset_path, val_transforms)

    print("Clustering completed.")

    # evaluate with ranking CMC on the validation set
    gallery_query_pairs = [
        # ('/home/oron_nir/clipreid/data/PRCC/prcc/rgb/test/A', '/home/oron_nir/clipreid/data/PRCC/prcc/rgb/test/B'),
        # ('/home/oron_nir/clipreid/data/PRCC/prcc/rgb/test/A', '/home/oron_nir/clipreid/data/PRCC/prcc/rgb/test/C'),
        # ('/home/oron_nir/clipreid/data/PRCC/prcc/rgb/test/B', '/home/oron_nir/clipreid/data/PRCC/prcc/rgb/test/A'),
        # ('/home/oron_nir/clipreid/data/PRCC/prcc/rgb/test/B', '/home/oron_nir/clipreid/data/PRCC/prcc/rgb/test/C'),
        # ('/home/oron_nir/clipreid/data/PRCC/prcc/rgb/test/C', '/home/oron_nir/clipreid/data/PRCC/prcc/rgb/test/A'),
        # ('/home/oron_nir/clipreid/data/PRCC/prcc/rgb/test/C', '/home/oron_nir/clipreid/data/PRCC/prcc/rgb/test/B'),
        # ('/home/oron_nir/clipreid/data/PRCC/prcc/rgb/train', '/home/oron_nir/clipreid/data/PRCC/prcc/rgb/val'),
        ('/home/oron_nir/clipreid/data/PRCC/prcc/rgb/val', '/home/oron_nir/clipreid/data/PRCC/prcc/rgb/train'),
    ]

    results = {}
    for gallery_dir, query_dir in gallery_query_pairs:
        print(f"Starting ranking CMC evaluation on gallery: {gallery_dir} and query: {query_dir}...")
        rank_1, rank_5, rank_10, mAP, n_Q = eval_with_ranking_cmc(model, gallery_dir, query_dir)
        logger.info(f"Ranking CMC evaluation results - Rank-1: {rank_1:.3%}, Rank-5: {rank_5:.3%}, Rank-10: {rank_10:.3%}, mAP: {mAP:.3%}, n_query: {n_Q}")
        print(f"Ranking CMC evaluation results - Rank-1: {rank_1:.3%}, Rank-5: {rank_5:.3%}, Rank-10: {rank_10:.3%}, mAP: {mAP:.3%}, n_query: {n_Q}")
        results[(os.path.basename(gallery_dir), os.path.basename(query_dir))] = {
            'rank_1': '{:.3%}'.format(rank_1),
            'rank_5': '{:.3%}'.format(rank_5),
            'rank_10': '{:.3%}'.format(rank_10),
            'mAP': '{:.3%}'.format(mAP),
            'n_query': n_Q,
            'n_gallery': len(os.listdir(gallery_dir))
        }

    print("Ranking CMC evaluation completed:\n{}".format(results))

