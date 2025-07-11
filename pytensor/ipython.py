import anywidget
import ipywidgets as widgets
import traitlets
from IPython.display import display

from pytensor.graph import FunctionGraph, Variable, rewrite_graph
from pytensor.graph.features import AlreadyThere, FullHistory


class CodeBlockWidget(anywidget.AnyWidget):
    """Widget that displays text content as a monospaced code block."""

    content = traitlets.Unicode("").tag(sync=True)

    _esm = """
    function render({ model, el }) {
      const pre = document.createElement("pre");
      pre.style.backgroundColor = "#f5f5f5";
      pre.style.padding = "10px";
      pre.style.borderRadius = "4px";
      pre.style.overflowX = "auto";
      pre.style.maxHeight = "500px";

      const code = document.createElement("code");
      code.textContent = model.get("content");

      pre.appendChild(code);
      el.appendChild(pre);

      model.on("change:content", () => {
        code.textContent = model.get("content");
      });
    }
    export default { render };
    """

    _css = """
    .jp-RenderedHTMLCommon pre {
      font-family: monospace;
      white-space: pre;
      line-height: 1.4;
    }
    """


class InteractiveRewrite:
    """
    Visualize a graph history through a series of rewrites.
    """

    def __init__(
        self,
        fg,
        display_reason=True,
        rewrite_options: dict | None = None,
        dprint_options: dict | None = None,
    ):
        """
        Parameters:
        -----------
        fg : FunctionGraph (or Variables)
            The function graph to track
        display_reason : bool, optional
            Whether to display the reason for each rewrite
        rewrite_options : dict, optional
            Options for rewriting the graph. Defaults to {'include': ('fast_run',), 'exclude': ('inplace',)}
        print_options : dict, optional
            Print options passed to `debugprint` used to generate the text representation of the graph.
            Useful options are {'print_shape': True, 'print_op_info': True}
        """
        self.dprint_options = dprint_options or {}
        self.rewrite_options = rewrite_options or dict(
            include=("fast_run",), exclude=("inplace",)
        )
        self.history = FullHistory(callback=self._history_callback)
        if not isinstance(fg, FunctionGraph):
            outs = [fg] if isinstance(fg, Variable) else fg
            fg = FunctionGraph(outputs=outs)
        try:
            fg.attach_feature(self.history)
        except AlreadyThere:
            self.history.end()

        self.updating_from_callback = False  # Flag to prevent recursion
        self.code_widget = CodeBlockWidget(content="")
        self.display_reason = display_reason

        if self.display_reason:
            self.reason_label = widgets.HTML(
                value="", description="", style={"description_width": "initial"}
            )
        self.slider_label = widgets.Label(value="")
        self.slider = widgets.IntSlider(
            value=self.history.pointer,
            min=0,
            max=0,
            step=1,
            description="",  # Empty description since we're using a separate label
            continuous_update=True,
            layout=widgets.Layout(width="300px"),
        )
        self.prev_button = widgets.Button(description="← Previous")
        self.next_button = widgets.Button(description="Next →")
        self.slider.observe(self._on_slider_change, names="value")
        self.prev_button.on_click(self._on_prev_click)
        self.next_button.on_click(self._on_next_click)

        self.rewrite_button = widgets.Button(
            description="Apply Rewrites",
            button_style="primary",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Apply default rewrites to the current graph",
            icon="cogs",  # Optional: add an icon (requires font-awesome)
        )
        self.rewrite_button.on_click(self._on_rewrite_click)

        self.nav_button_box = widgets.HBox([self.prev_button, self.next_button])
        self.slider_box = widgets.HBox([self.slider_label, self.slider])
        self.control_box = widgets.HBox([self.slider_box, self.rewrite_button])

        # Update the display with the initial state
        self._update_display()

    def _on_slider_change(self, change):
        """Handle slider value changes"""
        if change["name"] == "value" and not self.updating_from_callback:
            self.updating_from_callback = True
            index = change["new"]
            self.history.goto(index)
            self._update_display()
            self.updating_from_callback = False

    def _on_prev_click(self, b):
        """Go to previous history item"""
        if self.slider.value > 0:
            self.slider.value -= 1

    def _on_next_click(self, b):
        """Go to next history item"""
        if self.slider.value < self.slider.max:
            self.slider.value += 1

    def _on_rewrite_click(self, b):
        """Handle rewrite button click"""
        self.slider.value = self.slider.max
        self.rewrite()

    def display(self):
        """Display the full widget interface"""
        display(
            widgets.VBox(
                [
                    self.control_box,
                    self.nav_button_box,
                    *((self.reason_label,) if self.display_reason else ()),
                    self.code_widget,
                ]
            )
        )

    def _ipython_display_(self):
        self.display()

    def _history_callback(self):
        """Callback for history updates that prevents recursion"""
        if not self.updating_from_callback:
            self.updating_from_callback = True
            self._update_display()
            self.updating_from_callback = False

    def _update_display(self):
        """Update the code widget with the current graph and reason"""
        # Update the reason label if checkbox is checked
        if self.display_reason:
            if self.history.pointer == -1:
                reason = ""
            else:
                reason = self.history.fw[self.history.pointer].reason
                reason = getattr(reason, "name", None) or str(reason)

            self.reason_label.value = f"""
                <div style='padding: 5px; margin-bottom: 10px; background-color: #e6f7ff; border-left: 4px solid #1890ff;'>
                    <b>Rewrite:</b> {reason}
                </div>
            """

        # Update the graph display
        self.code_widget.content = self.history.fg.dprint(
            file="str", **self.dprint_options
        )

        # Update slider range if history length has changed
        history_len = len(self.history.fw) + 1
        if history_len != self.slider.max + 1:
            self.slider.max = history_len - 1

        # Update slider value without triggering the observer
        if not self.updating_from_callback:
            with self.slider.hold_trait_notifications():
                self.slider.value = self.history.pointer + 1

        # Update the slider label to show current position and total (1-based)
        self.slider_label.value = (
            f"History: {self.history.pointer + 1}/{history_len - 1}"
        )

    def rewrite(self, *args, **kwargs):
        """Apply rewrites to the current graph"""
        rewrite_graph(
            self.history.fg,
            *args,
            **kwargs,
            **self.rewrite_options,
            clone=False,
        )
        self._update_display()
