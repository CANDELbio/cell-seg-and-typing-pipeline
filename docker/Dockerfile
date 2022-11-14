FROM continuumio/miniconda3:4.10.3
RUN conda install --yes -c conda-forge tifffile
RUN conda install --yes -c conda-forge scikit-image
RUN conda install --yes -c conda-forge ome-types
RUN conda install --yes -c conda-forge pyvips
RUN conda install --yes numpy
RUN conda install --yes pandas
RUN conda install --yes scipy
RUN conda install --yes scikit-learn
RUN conda install --yes openpyxl
ADD collapse.py /collapse.py
ADD listchannels.py /listchannels.py
ADD segment_summary.py /segment_summary.py
ADD bootstrap_mibi_pixels.py /bootstrap_mibi_pixels.py
ADD pca_mibi.py /pca_mibi.py
ADD type_segments.py /type_segments.py
ADD expand_segmentation.py /expand_segmentation.py
ADD tile.py /tile.py
