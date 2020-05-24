import numpy
import matplotlib
matplotlib.use("AGG")
from matplotlib import pyplot
from google.cloud import storage


def upload_blob(source_file_name, destination_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket("copymonkey")
    destination_name = "static/uploads/" + destination_name
    blob = bucket.blob(destination_name)

    blob.upload_from_filename(source_file_name)

    print("File {} uploaded to {}.".format(source_file_name, destination_name))


def plot_stroke(stroke, save_name=None):
    # Plot a single example.
    f, ax = pyplot.subplots()

    x = numpy.cumsum(stroke[:, 1])
    y = numpy.cumsum(stroke[:, 2])

    size_x = x.max() - x.min() + 1.0
    size_y = y.max() - y.min() + 1.0

    f.set_size_inches(5.0 * size_x / size_y, 5.0)

    cuts = numpy.where(stroke[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value], "k-", linewidth=3)
        start = cut_value + 1

    ax.axis("off")  # equal
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        pyplot.show()
    else:
        try:
            pyplot.savefig(save_name, bbox_inches="tight", pad_inches=0.5)
        except Exception:
            print("Error building image!: " + save_name)

    pyplot.close()
