from functions.visualize import plot_grid_samples_tensor


def log_to_tensorboard(writer, images_to_log, metrics_to_log, i):
    
    for name, image in images_to_log.items():
        grid = plot_grid_samples_tensor(image, grid_size=[4,4])
        writer.add_image(name, grid, i)
    
    for name, metric in metrics_to_log.items():
        writer.add_scalar(name, metric, i)
        