from tensorboard import summary
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorflow.contrib.learn.python.learn.summary_writer_cache import SummaryWriterCache


def add_custom_scalar(logdir):
    summary_writer = SummaryWriterCache.get(logdir)
    layout_summary = summary.custom_scalar_pb(layout_pb2.Layout(
        category=[
            layout_pb2.Category(
                title='Loss',
                chart=[
                    layout_pb2.Chart(
                        title='Loss',
                        multiline=layout_pb2.MultilineChartContent(
                            tag=[r'1_loss/*'],
                        )),
                    layout_pb2.Chart(
                        title='Loss Component',
                        multiline=layout_pb2.MultilineChartContent(
                            tag=[r'2_loss_component/*'],
                        )),
                    layout_pb2.Chart(
                        title='Discriminator Values',
                        multiline=layout_pb2.MultilineChartContent(
                            tag=[r'3_discriminator_values/*'],
                        )),
                    layout_pb2.Chart(
                        title='Variation of sequences',
                        multiline=layout_pb2.MultilineChartContent(
                            tag=[r'Stddev/*'],
                        )),
                    layout_pb2.Chart(
                        title='BLOMSUM45',
                        multiline=layout_pb2.MultilineChartContent(
                            tag=[r'Blast/*/BLOMSUM45'],
                        )),
                    layout_pb2.Chart(
                        title='Evalue',
                        multiline=layout_pb2.MultilineChartContent(
                            tag=[r'Blast/*/Evalue'],
                        )),
                    layout_pb2.Chart(
                        title='Identity',
                        multiline=layout_pb2.MultilineChartContent(
                            tag=[r'Blast/*/Identity'],
                        )),
                ]),
        ]))
    summary_writer.add_summary(layout_summary)