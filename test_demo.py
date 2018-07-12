from east_funcs import *
from compteur_funcs import *
import matplotlib.patches as patches

# Load east model
east_checkpoint_path = 'east_icdar2015_resnet_v1_50_rbox/'
sess_east, input_images, f_score, f_geometry = load_model(east_checkpoint_path)
# Load mnist model
mnist_checkpoint_path = 'model_mnist/model_saved-900'
sess_mnist, x, y_pred, y_pred_cls = load_mnist_model(mnist_checkpoint_path)

def main(image_path, output_path):
    image_ori, image_resized = preprocess_image(image_path, max_side_len=700)
    dictboxes, score, geome = detectboxes(image_resized,
                            east_session=sess_east,
                            image_placeholder=input_images,
                            f_score_placeholder=f_score, f_geometry_placeholder=f_geometry)
    dictcropims = crop_image(image_ori, image_resized, dictboxes, margin=10)
    good_zoi = select_good_box2(dictboxes, dictcropims)
    # Detection
    snipets, regions, postpro_im = preprocess_zoi2(good_zoi, val_precenti=95, blob_min_size_coeff=2, blob_max_size_coeff=12, margin=15)
    preds, pred_labels = predict_snippets(snipets, sess_mnist, x, y_pred, y_pred_cls)
    plt.figure()
    # Create figure and axes
    fig,ax = plt.subplots(1)
    # Display the image
    ax.imshow(good_zoi)
    for i, re in enumerate(regions):
        box_example = re.bbox
        width = int(box_example[2] - box_example[0])
        height = int(box_example[3] - box_example[1])
        begining_rect = (box_example[1], box_example[0])
        cx = min(max(begining_rect[0] + 5, 0), good_zoi.shape[1] - 1)
        cy = min(max(begining_rect[1] - 5, 0), good_zoi.shape[0] - 1)
        ax.add_patch(patches.Rectangle(begining_rect, height, width, linewidth=1,edgecolor='r',facecolor='none'))
        ax.annotate(str(int(pred_labels[i])), (cx, cy), color='g', weight='bold', fontsize=20, ha='center', va='center')
    plt.show()
    fig.savefig(output_path)


if __name__ == '__main__':
    main('tes_compteur_im.jpg', 'tes_compteur_im_output.jpg')
