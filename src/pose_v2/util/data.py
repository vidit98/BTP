import tensorflow as tf


def tf_record_options(cfg):
    compression_type = tf.compat.v1.python_io.TFRecordCompressionType
    if cfg.tfrecords_gzip_compressed:
        compression = compression_type.GZIP
    else:
        compression = compression_type.NONE
    return tf.io.TFRecordOptions(compression)


def tf_record_compression(cfg):
    if cfg.tfrecords_gzip_compressed:
        compression = "GZIP"
    else:
        compression = ""
    return compression