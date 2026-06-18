# ================================================================
# 0. Section: IMPORTS
# ================================================================
from copy import deepcopy

from clearbrain.domain_model.data import TissueType
from clearbrain.service.ClearTissueProject import ClearTissueProject
import clearbrain.domain_model.transformations as tr

from clearbrain.registration import Registrator, RigidRegistration, RegistratorResampler, RegistrationConfig



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    """ project = ClearTissueProject.init(
        mouse="32B",
        tissue_type=TissueType.SPINAL_CORD,
    ) """
    project = ClearTissueProject.load(
        mouse="32B",
        tissue_type=TissueType.SPINAL_CORD,
    )
    raw_batch = project.load_raw()

    pipeline = project.init_pipeline()

    tissue_registrator = Registrator(
        strategy=RigidRegistration(),
        resampler=RegistratorResampler(),
        config=RegistrationConfig(),
    )
    tissue_registrator.config.metric.name = "CC"
    tissue_registrator.config.metric.sampling_percentage = 1
    tissue_registrator.config.metric.histogram_bins = 50
    tissue_registrator.config.optimizer.iterations = 500
    tissue_registrator.config.optimizer.convergence_minimum_value = 1e-8
    tissue_registrator.config.optimizer.convergence_window_size = 30

    cell_registrator = deepcopy(tissue_registrator)
    cell_registrator.config.interpolator.resampling = "nearest"

    pipeline.add_list([
        tr.RegularizeSample(),
        tr.OrientSample(),
        tr.StretchSample(smooth_window_size=25),
        tr.UntwistSample(
            tissue_registrator=tissue_registrator,
            cell_registrator=cell_registrator,
            window_size=250,
            gap_size=0
        )
    ])

    final_batch = project.run_pipeline(pipeline, raw_batch)
