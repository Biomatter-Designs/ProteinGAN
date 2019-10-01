from gan.protein.helpers import FAKE_PROTEINS
from gan.protein.protein import LABELS
from tensorflow.contrib.learn.python.learn.summary_writer_cache import SummaryWriterCache
from tensorflow.python import ops
from tensorflow.python.training import session_run_hook, training_util
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorflow.python.training.session_run_hook import SessionRunArgs


class BlastHook(session_run_hook.SessionRunHook):
    """Hook that counts steps per second."""

    def __init__(self,
                 summary,
                 config,
                 id_to_enzyme_class,
                 every_n_steps=1200,
                 every_n_secs=None,
                 output_dir=None,
                 summary_writer=None,
                 n_examples=2,
                 running_mode="train"):

        self._timer = SecondOrStepTimer(every_steps=every_n_steps, every_secs=every_n_secs)
        self.summary = summary
        self.config = config
        self.summary_writer = summary_writer
        self.output_dir = output_dir
        self.last_global_step = None
        self.id_to_enzyme_class = id_to_enzyme_class
        self.global_step_check_count = 0
        self.steps_per_run = 1
        self.n_examples = n_examples,
        self.running_mode = running_mode

    def _set_steps_per_run(self, steps_per_run):
        self.steps_per_run = steps_per_run

    def begin(self):
        if self.summary_writer is None and self.output_dir:
            self.summary_writer = SummaryWriterCache.get(self.output_dir)
        graph = ops.get_default_graph()
        self.fake_seq = graph.get_tensor_by_name("model/" + FAKE_PROTEINS + ":0")
        self.labels = graph.get_tensor_by_name("model/" + LABELS + ":0")
        self.d_score = graph.get_tensor_by_name("model/d_score:0")
        self.global_step_tensor = training_util._get_or_create_global_step_read()
        if self.global_step_tensor is None:
            raise RuntimeError("Could not global step tensor")
        if self.fake_seq is None:
            raise RuntimeError("Could not get fake seq tensor")

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return SessionRunArgs([self.global_step_tensor, self.fake_seq, self.labels, self.d_score])

    def after_run(self, run_context, run_values):
        global_step, fake_seq, labels, d_score = run_values.results
        if self._timer.should_trigger_for_step(global_step):
            # fake_seq, real_seq, labels = run_context.session.run([self._fake_seq, self._real_seq, self._labels])
            self.summary(self.config, self.summary_writer, global_step, fake_seq, labels, self.id_to_enzyme_class,
                         self.n_examples[0], self.running_mode, d_score).start()
            self._timer.update_last_triggered_step(global_step)
