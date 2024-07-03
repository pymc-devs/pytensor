import * as d3 from "https://cdn.skypack.dev/pin/d3@v7.8.5-eC7TKxlFLay7fmsv0gvu/dist=es2020,mode=imports,min/optimized/d3.js";
import "https://cdn.skypack.dev/-/d3-graphviz@v5.1.0-TcGnvMu4khUzCpL7Wr2k/dist=es2020,mode=imports,min/optimized/d3-graphviz.js";

export function render({ model, el }) {
  if (!document.getElementById("graphviz_script")) {
    const graphviz_script = document.createElement("script");
    graphviz_script.setAttribute("id", "graphviz_script");
    graphviz_script.setAttribute("src", "https://unpkg.com/@hpcc-js/wasm/dist/graphviz.umd.js");
    graphviz_script.setAttribute("type", "javascript/worker");

    document.head.appendChild(graphviz_script);
  }
  //   let getCount = () => model.get("dot");

  const div = document.createElement("div");
  div.classList.add("graphviz-container");
  const graphContainer = d3.select(div);

  let setDot = () => {
    const dots = model.get("dots");
    const performance = model.get("performance");
    // Use the last dots if there is no index
    const dot = dots[model.get("index")];
    // const width = div.clientWidth;
    // const height = div.clientHeight;
    const graphviz = graphContainer
      .graphviz({
        // Fit graph to that size, so that all is visible
        fit: true,
        // Set to be as big as container
        // width,
        // height,
        // Don't animate transitions between shapes for performance
        tweenPaths: !performance,
        tweenShapes: !performance,
        useWorker: true,
      })
      .transition(() => d3.transition("t").duration(2000).ease(d3.easeLinear))
      .renderDot(dot);
    // If we have made a zoom selection, reset that before transitioning
    // TODO: figure out how to transition BOTH zoom and dot at once
    // if (graphviz._zoomSelection) {
    //   graphviz.resetZoom();
    // }
  };

  model.on("change:dots change:index", setDot);
  el.appendChild(div);
  requestAnimationFrame(() => setTimeout(setDot, 0));
}
