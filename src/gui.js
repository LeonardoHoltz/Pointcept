import { GUI } from 'three/addons/libs/lil-gui.module.min.js';
import { render } from "./rendering.js";
import configData from '../config/conf.json' assert { type: 'json' };
import { buildLegend } from "./legend.js";

import './styles/gui.css';

/**
 * Inicializa a GUI para um cloud e opções fornecidas
 * @param {THREE.Points} cloud - nuvem de pontos
 * @param {object} options - { colorBy, colormap, pointSize, labelsVisibility }
 */
export function initGUI(pointCloud, options) {
    const gui = new GUI(
        {
            title: "Point Cloud",
            width: 900
        }
    );
    gui.domElement.classList.add('pc-gui'); // Uses gui.css
    
    // --- Section: Cloud common visualization options ---
    const cloudGeneral = gui.addFolder('Main options');
    cloudGeneral.open();
    
    // Show/Hide Offset Point Cloud
    cloudGeneral.add(options, 'show_pc').name('Show Point Cloud').onChange((value) => {
        pointCloud.visible = value;
        render();
    });

    // How the Point Cloud should be colored
    cloudGeneral.add(options, 'colorBy', ['rgb', 'semantic_gt', 'semantic_pred', 'confidences', 'density'])
        .name('Color by')
        .onChange(async () => {
            pointCloud.recolor(options.colorBy, options.colormap);
            if (options.clustering_mode)
                pointCloud.prepareClustering(true, configData.classes.indexOf(options.label_select));
            render();
        });
    
    // When not using rgb, which type of colormap to use
    cloudGeneral.add(options, 'colormap', ['Category10', 'Set3', 'Paired', 'viridis', 'plasma'])
        .name('Colormap')
        .onChange(async () => {
            if (['Category10', 'Set3', 'Paired'].includes(options.colormap))
                buildLegend(options.colormap_mapping[options.colormap]);
            pointCloud.recolor(options.colorBy, options.colormap);
            if (options.clustering_mode)
                pointCloud.prepareClustering(true, configData.classes.indexOf(options.label_select));
            render();
        });

    // Size of the points
    cloudGeneral.add(options, 'pointSize', 1, 10, 0.1)
        .name('Point Size')
        .onChange(() => {
            pointCloud.changeSize(options.pointSize);
            render();
        });

    // Slider to check offset progression compared with main PC
    cloudGeneral.add(options, 'offset_pred_slider', 0, 1).name('Offset progression').onChange((value) => {
        pointCloud.applyOffset(value);
        render();
    });

    // Unimportant Labels visibility
    cloudGeneral.add(options, 'hide_labels').name('Hide non-equipment objects').onChange((value) => {
        pointCloud.updateLabelVisibility(value);
        render();
    });

    // --- Section: Clustering Mode ---
    const clusteringFolder = gui.addFolder('Clustering options');
    clusteringFolder.add(options, 'clustering_mode').name('Clustering mode').onChange((value) => {
        pointCloud.prepareClustering(value, configData.classes.indexOf(options.label_select));
        render();
    });
    clusteringFolder.add(options, 'label_select', configData.classes).name('Select label').onChange((value) => {
        options.reset_clustering();
    });
    const algorithm_options = ['DBSCAN', 'Ball-Query'];
    let algorithmParameters;
    let parametersControllers = [];
    const algorithm = clusteringFolder.add(options, 'algorithm', algorithm_options).name('Clustering Algorithm').onChange((value) => {
        while(parametersControllers.length > 0) {
            const controller = parametersControllers.pop();
            controller.destroy();
        };
        switch(value) {
            case 'DBSCAN':
                algorithmParameters.title(options.algorithm_parameters.dbscan.title);
                parametersControllers.push(
                    algorithmParameters.add(options.algorithm_parameters.dbscan, 'eps').name('eps (distance)')
                );
                parametersControllers.push(
                    algorithmParameters.add(options.algorithm_parameters.dbscan, 'min_pts').name('Min. Points belonging to cluster')
                );
                break;
            case 'Ball-Query':
                algorithmParameters.title(options.algorithm_parameters.ball_query.title);
                parametersControllers.push(
                    algorithmParameters.add(options.algorithm_parameters.ball_query, 'radius').name('Radius')
                );
                break;
        };
    });
    algorithmParameters = clusteringFolder.addFolder('placeholder');
    algorithm.setValue(algorithm_options[0]); // Force the onchange in the first time

    clusteringFolder.add(options, 'apply_clustering').name('Apply Clustering');
    clusteringFolder.add(options, 'reset_clustering').name('Reset Clustering');

    return gui;
}