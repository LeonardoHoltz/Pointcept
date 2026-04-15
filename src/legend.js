import configData from '../config/conf.json' assert { type: 'json' };

export function buildLegend(color_scheme) {
    const classColors = {};
    configData.classes.forEach((item, i) => {
        classColors[item] = color_scheme[i % 10];
    });
    console.log("Legend colors:", classColors);
    console.log(color_scheme);
    const legend = document.getElementById("legend");
    
    // Removes everything except the title
    while (legend.children.length > 1) {
        legend.removeChild(legend.lastChild);
    }

    for (const [label, color] of Object.entries(classColors)) {
        const item = document.createElement("div");
        item.className = "legend-item";

        const colorBox = document.createElement("div");
        colorBox.className = "legend-color";
        colorBox.style.background = color;

        const text = document.createElement("span");
        text.innerText = label;

        item.appendChild(colorBox);
        item.appendChild(text);
        legend.appendChild(item);
    }
}