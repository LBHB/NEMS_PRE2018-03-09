import boto3
from sqlalchemy.sql import not_

import nems_config.Storage_Config as sc
from nems.db import Session, tQueue, tComputer


def check_instance_count():
    session = Session()
    # TODO: How to check which jobs are currently running/waiting?
    num_jobs = (
        session.query(tQueue)
        .filter(tQueue.progress == 'something')
        .count()
    )

    ec2 = boto3.resources('ec2')
    instances = ec2.instances.filter(
        Filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
    )
    run_count = len(instances)

    if num_jobs / run_count > awsc.JOBS_PER_INSTANCE:
        # TODO: Where to get image id?
        ec2.create_instances(ImageId='<ami-image-id>', MinCount=1, MaxCount=5)
    elif num_jobs / run_count - 1 < sc.JOBS_PER_INSTANCE:
        ids = (
            session.query(tComputer)
            .filter(not_(tComputer.name.in_(awsc.LOCAL_MACHINES)))
        )
        # Could also filter based on this criteria in query maybe
        ids.remove('remove ids based on some criteria - dont stop them all')
        ec2.instances.filter(InstanceIds=ids).stop()
        ec2.instances.filter(InstanceIds=ids).terminate()
