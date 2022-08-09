import os
import cv2
import sys
import torch
from tqdm import tnrange
import matplotlib.pyplot as plt
from adaptis.inference.adaptis_sampling import get_panoptic_segmentation
from adaptis.inference.prediction_model import AdaptISPrediction
from adaptis.data.toy import ToyDataset
from adaptis.model.toy.models import get_unet_model
from adaptis.coco.panoptic_metric import PQStat, pq_compute, print_pq_stat
from adaptis.utils.vis import visualize_instances, visualize_proposals


sys.path.insert(0, '..')


def dataset_model(dataset_path, weights_path, num_classes):
    dataset = ToyDataset(dataset_path, split='test_samples', with_segmentation=True)

    model = get_unet_model(norm_layer=torch.nn.BatchNorm2d, num_classes=num_classes, with_proposals=True)
    #model = get_unet_model(norm_layer=torch.nn.BatchNorm2d, with_proposals=True)
    
    pmodel = AdaptISPrediction(model, dataset, device)
    pmodel.load_parameters(weights_path)

    return pmodel, dataset


def test_model(pmodel, dataset,
               sampling_algorithm, sampling_params,
               use_flip=False, cut_radius=-1):
    pq_stat = PQStat()
    categories = dataset._generate_coco_categories()
    categories = {x['id']: x for x in categories}

    for indx in range(len(dataset)):
        sample = dataset.get_sample(indx)
        pred = get_panoptic_segmentation(pmodel, sample['image'],
                                         sampling_algorithm=sampling_algorithm,
                                         use_flip=use_flip, cut_radius=cut_radius, **sampling_params)
        
        
        coco_sample = dataset.convert_to_coco_format(sample)
        pred = dataset.convert_to_coco_format(pred)

        pq_stat = pq_compute(pq_stat, pred, coco_sample, categories)
    
    print_pq_stat(pq_stat, categories)


def panoptic_seg(dataset, pmodel, vis_samples, proposals_sampling_params, fig_path):
    fig, ax = plt.subplots(nrows=len(vis_samples), ncols=3, figsize=(7,7))
    fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)    
    
    for row_indx, sample_indx in enumerate(vis_samples):
        sample = dataset.get_sample(sample_indx)
        pred = get_panoptic_segmentation(pmodel, sample['image'],
                                    sampling_algorithm='proposals',
                                    use_flip=True, **proposals_sampling_params)

        for i in range(3):
            ax[row_indx, i].axis('off')

        if row_indx == 0:
            ax[row_indx, 0].set_title('Input Image', fontsize=14)
            ax[row_indx, 1].set_title('Instance Segmentation', fontsize=14)
            ax[row_indx, 2].set_title('Proposal Map', fontsize=14)
        ax[row_indx, 0].imshow(sample['image'])
        ax[row_indx, 1].imshow(visualize_instances(pred['instances_mask']))
        ax[row_indx, 2].imshow(visualize_proposals(pred['proposals_info']))

        fig.savefig(fig_path)
        plt.close(fig)


def panoptic_seg_dense(sample_image_path, pmodel, dense_sampling_params, instance_mask_path, prediction_path):
    sample_image = cv2.imread(sample_image_path)[:, :, ::-1].copy()
    plt.figure(figsize=(8, 8))
    #plt.imshow(sample_image)

    pred = get_panoptic_segmentation(pmodel, sample_image,
                                    sampling_algorithm='proposals',
                                    use_flip=True, **dense_sampling_params)

    plt.figure(figsize=(8,8))
    result = visualize_instances(pred['instances_mask'],
                                   boundaries_color=(150, 150, 150), boundaries_alpha=0.8)
    plt.imsave(instance_mask_path, result)
    plt.close()

    plt.figure(figsize=(8,8))
    result = visualize_proposals(pred['proposals_info'])
    plt.imsave(prediction_path, result)
    plt.close()


if __name__ == '__main__':
    # Set the device, dataset path and weights path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
    dataset_path = os.path.join('app', 'custom_dataset')
    weights_path = os.path.join('experiments', 'toy_v2', '132',
    'checkpoints', 'proposals_last_checkpoint.params') 
    num_classes = 5
 
    # Get the model and dataset
    pmodel, dataset = dataset_model(dataset_path, weights_path, num_classes)

    # Test proposals-based point sampling and random sampling
    proposals_sampling_params = {
        'thresh1': 0.4,
        'thresh2': 0.5,
        'ithresh': 0.3,
        'fl_prob': 0.10,
        'fl_eps': 0.003,
        'fl_blur': 2,
        'max_iters': 100
    }

    random_sampling_params = {
        'thresh1': 0.4,
        'thresh2': 0.5,
        'ithresh': 0.3,
        'num_candidates': 7,
        'num_iters': 40
    }

    print('#---------------Test proposals-based point sampling---------------#')
    test_model(pmodel, dataset,
            sampling_algorithm='proposals',
            sampling_params=proposals_sampling_params,
            use_flip=False)

    test_model(pmodel, dataset,
            sampling_algorithm='proposals',
            sampling_params=proposals_sampling_params,
            use_flip=True)

    print('#---------------Test random sampling---------------#')
    test_model(pmodel, dataset,
            sampling_algorithm='random', sampling_params=random_sampling_params,
            use_flip=False)

    test_model(pmodel, dataset,
            sampling_algorithm='random', sampling_params=random_sampling_params,
            use_flip=True)
    
    # Results visualization
    vis_samples = [0, 1, 2]

    fig_path = os.path.join('app', 'testing_plots', 'panoptic_seg_results.png')

    panoptic_seg(dataset, pmodel, vis_samples, proposals_sampling_params, fig_path)

    # Test challenging samples
    # dense_sampling_params = {
    # 'thresh1': 0.75,
    # 'thresh2': 0.50,
    # 'ithresh': 0.3,
    # 'fl_prob': 0.10,
    # 'fl_eps': 0.003,
    # 'fl_blur': 2,
    # 'max_iters': 1000,
    # 'cut_radius': 48
    # }

    # sample_image_path = os.path.join('app', 'custom_dataset', 'test_samples', 'stn1_syn007_pkg000_0_89_rep_rgb.png')
    # instance_mask_path = os.path.join('app', 'testing_plots', 'pred_instance_mask.png')
    # prediction_path = os.path.join('app', 'testing_plots', 'pred_proposal_map.png')

    # panoptic_seg_dense(sample_image_path, pmodel, dense_sampling_params, instance_mask_path, prediction_path)
    

