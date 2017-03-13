import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import vgg16
import os


#helper function for images
def load_image(filename, max_size=None):
    image = PIL.Image.open(filename)

    if max_size is not None:
        # Calculate the appropriate rescale-factor for
        # ensuring a max height and width, while keeping
        # the proportion between them.
        factor = max_size / np.max(image.size)

        # Scale the image's height and width.
        size = np.array(image.size) * factor

        # The size is now floating-point because it was scaled.
        # But PIL requires the size to be integers.
        size = size.astype(int)

        # Resize the image.
        image = image.resize(size, PIL.Image.LANCZOS)

    # Convert to numpy floating-point array.
    return np.float32(image)

def save_image(image, filename):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)

    # Convert to bytes.
    image = image.astype(np.uint8)

    # Write the image-file in jpeg-format.
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')



def plot_image_big(image):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)

    # Convert pixels to bytes.
    image = image.astype(np.uint8)

    # Convert to a PIL-image and display it.
    display(PIL.Image.fromarray(image))

def plot_images(content_image, style_image, mixed_image):
    # Create figure with sub-plots.
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Use interpolation to smooth pixels?
    smooth = True

    # Interpolation type.
    if smooth:
        interpolation = 'sinc'
    else:
        interpolation = 'nearest'

    # Plot the content-image.
    # Note that the pixel-values are normalized to
    # the [0.0, 1.0] range by dividing with 255.
    ax = axes.flat[0]
    ax.imshow(content_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Content")

    # Plot the mixed-image.
    ax = axes.flat[1]
    ax.imshow(mixed_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Mixed")

    # Plot the style-image
    ax = axes.flat[2]
    ax.imshow(style_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Style")

    # Remove ticks from all the plots.
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def mean_squared_error(a,b):
    return tf.reduce_mean(tf.square(a-b))

def create_content_loss(session,model,content_image,layer_ids):
    feed_dict = model.create_feed_dict(image=content_image)
    layers = model.get_layer_tensors(layer_ids)
    values = session.run(layers,feed_dict=feed_dict)

    with model.graph.as_default():
        layer_losses= []
        for value,layer in zip(values,layers):
             value_const = tf.constant(value)
             loss = mean_squared_error(layer,value_const)
             layer_losses.append(loss)
        total_loss = tf.reduce_mean(layer_losses)
    return total_loss

def gram_matrix(tensor):
    shape = tensor.get_shape()
    num_channels = int(shape[3])
    matrix = tf.reshape(tensor,[-1,num_channels])
    gram = tf.matmul(tf.transpose(matrix),matrix)
    return gram

def create_style_loss(session,model,style_image,layer_ids):
    feed_dict = model.create_feed_dict(image=style_image)
    layers = model.get_layer_tensors(layer_ids)

    with model.graph.as_default():
        gram_layers =[gram_matrix(layer) for layer in layers]
        values = session.run(gram_layers,feed_dict=feed_dict)
        layer_losses = []
        for value,gram_layer in zip(values,gram_layers):
            value_const = tf.constant(value)
            loss = mean_squared_error(gram_layer,value_const)
            layer_losses.append(loss)
            total_loss=tf.reduce_mean(layer_losses)
    return total_loss

def create_denoise_loss(model):
    loss = tf.reduce_sum(tf.abs(model.input[:,1:,:,:] - model.input[:,:-1,:,:])) + \
           tf.reduce_sum(tf.abs(model.input[:,:,1:,:] - model.input[:,:,:-1,:]))
    return loss


def style_transfer(content_image,style_image,content_layer_ids,style_layer_ids,weight_content=1.5,weight_style=10,weight_denoise =0.3,number_iterations =120,step_size =10):
    model = vgg16.VGG16()
    session = tf.InteractiveSession(graph = model.graph)
    # Create the loss-function for the content-layers and -image.
    loss_content = create_content_loss(session=session,
                                       model=model,
                                       content_image=content_image,
                                       layer_ids=content_layer_ids)

    # Create the loss-function for the style-layers and -image.
    loss_style = create_style_loss(session=session,
                                   model=model,
                                   style_image=style_image,
                                   layer_ids=style_layer_ids)

    # Create the loss-function for the denoising of the mixed-image.
    loss_denoise = create_denoise_loss(model)

    adj_content = tf.Variable(1e-10, name='adj_content')
    adj_style = tf.Variable(1e-10, name='adj_style')
    adj_denoise = tf.Variable(1e-10, name='adj_denoise')

    session.run([adj_content.initializer,
                 adj_style.initializer,
                 adj_denoise.initializer])

    update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
    update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))
    update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))

    loss_combined = weight_content*adj_content*loss_content + weight_style*adj_style*loss_style +weight_denoise*adj_denoise*loss_denoise

    gradient = tf.gradients(loss_combined,model.input)

    run_list = [gradient,update_adj_style,update_adj_denoise,update_adj_content]

    mixed_image = np.random.rand(*content_image.shape) + 128

    for i in range(number_iterations):
        feed_dict = model.create_feed_dict(image = mixed_image)
        gradient, adj_content_val, adj_style_val, adj_denoise_val = session.run(run_list,feed_dict=feed_dict)
        grad = np.squeeze(gradient)
        step_size_scaled = step_size/(np.std(grad) +1e-10)
        mixed_image -= grad*step_size_scaled
        mixed_image = np.clip(mixed_image,0.0,255.0)

        if(i%10 ==0) or (i == number_iterations -1):
           print("iterations",i)
           msg = "Weight Adj. for Content: {0:.2e}, Style: {1:.2e}, Denoise: {2:.2e}"
           print(msg.format(adj_content_val, adj_style_val, adj_denoise_val))
           plot_images(content_image=content_image,
                        style_image=style_image,
                        mixed_image=mixed_image)

    print("final image")
    plot_image_big(mixed_image)
    session.close()

    return mixed_image

def main():
    content_filename = 'images/content.jpg'
    content_image = load_image(content_filename, max_size=None)
    style_filename = 'images/style.jpg'
    style_image = load_image(style_filename, max_size=None)
    content_layer_ids = [4]
    style_layer_ids =[8,9,10]
    style_transfer(content_image,style_image,content_layer_ids,style_layer_ids,weight_content=1.5,weight_style=10,weight_denoise =0.3,number_iterations =120,step_size =10)


if __name__== "__main__":
    main()
