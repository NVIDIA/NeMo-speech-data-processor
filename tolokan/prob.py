import datetime
import logging
import sys

import IPython.display as display
import toloka.client as toloka
import toloka.client.project.template_builder as tb

logging.basicConfig(
    format='[%(levelname)s] %(name)s: %(message)s',
    level=logging.INFO,
    stream=sys.stdout,
)

# API_KEY2 = "JlyFMMVXFqhRzGnZ_WRclA.045e.1yv4VJjxRwFYS5ErLKaKGFkavejelS_h-Sq0Ey3A_cw37xVNK2zH3LRII32rAb4OwZv3pQQAhQCiB52_TKB3H6D4dDnwLD3aM3dX05d0wUE"
API_KEY3 = "1V0v8tbFtwPhDVMbN3cvcQ.045e.k93FVRUudCOHplT6OHcDNjc-k8V_rSFr8l5_jLyo4rLT9W4fgabM40EyompNakPZ0uU-_sgc0hKHzrEge5AJNy5MwZCu93BHoTJGOS1lwqE"

toloka_client = toloka.TolokaClient(API_KEY3, 'PRODUCTION')  # Or switch to 'SANDBOX'
# getpass.getpass('Enter your OAuth token: ')
# Lines below check that the OAuth token is correct and print your account's name
print(toloka_client.get_requester())


# new_project = toloka.Project(
#     public_name='Voice recording GOSHA',
#     public_description='Tap the voice recorder button and read the text aloud.',
# )

# text_view = tb.TextViewV1(tb.InputData('text'))
# audio_field = tb.AudioFieldV1(tb.OutputData('audio_file'), validation=tb.RequiredConditionV1())
# width_plugin = tb.TolokaPluginV1('scroll', task_width=500)

# project_interface = toloka.project.TemplateBuilderViewSpec(
#     view=tb.ListViewV1(items=[text_view, audio_field]),
#     plugins=[width_plugin]
# )

# input_specification = {'text': toloka.project.StringSpec()}
# output_specification = {'audio_file': toloka.project.FileSpec()}

# new_project.task_spec = toloka.project.task_spec.TaskSpec(
#         input_spec=input_specification,
#         output_spec=output_specification,
#         view_spec=project_interface,
# )

# new_project.public_instructions = """Each task contains words and phrases. You need to read and record them.
# Make a new recording for each phrase, even if it has already been used in other tasks."""

# new_project = toloka_client.create_project(new_project)

# new_pool = toloka.Pool(
#     project_id=new_project.id,
#     private_name='Voice recording',
#     may_contain_adult_content=False,
#     will_expire=datetime.datetime.utcnow() + datetime.timedelta(days=365),
#     reward_per_assignment=0.01,
#     assignment_max_duration_seconds=60*10,
#     auto_accept_solutions=False,
#     auto_accept_period_day=1,
#     filter=(
#         (toloka.filter.Languages.in_('EN')) &
#         (toloka.filter.ClientType == 'TOLOKA_APP')
#     ),
# )

# new_pool.set_mixer_config(real_tasks_count=5)
# new_pool = toloka_client.create_pool(new_pool)


# tasks = [
#         toloka.Task(input_values={'text': "дис из май тест"}, pool_id=new_pool.id)
#     ]

# toloka_client.create_tasks(tasks, allow_defaults=True)

# new_pool = toloka_client.open_pool(new_pool.id)


# for attachment in toloka_client.get_attachments(pool_id='42757899'):
#     print(attachment.id, attachment.name)

# print(attachment)

for assignment in toloka_client.get_assignments(pool_id='42757899'):
    # if assignment.status != "SKIPPED":
    #     task_text = assignment.tasks.input_values['text']

    #     attachment_id = assignment.solutions.output_values.get('audio_file', 'No attachment ID found')

    #     print(f"Task Text: {task_text}, Attachment ID: {attachment_id}")
    print("-------------------------------")
    print(assignment)
