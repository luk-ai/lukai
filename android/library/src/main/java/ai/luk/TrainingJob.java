package ai.luk;

import android.app.job.JobInfo;
import android.app.job.JobScheduler;
import android.app.job.JobService;
import android.app.job.JobParameters;
import android.content.Context;
import android.content.ComponentName;

import ai.luk.ModelType;

/*
To use this training job you'll need to extend it and implement the domain,
modelType and dataDir methods. Your code needs to call the schedule() method at
least once.
You'll also need to define the service in your AndroidManifest.xml file.

<service
android:name=".yourpackage.TrainingJob"
android:permission="android.permission.BIND_JOB_SERVICE"
android:permission="android.permission.RECEIVE_BOOT_COMPLETED"
android:exported="true"/>

*/
public abstract class TrainingJob extends JobService {
  private ModelType mt;

  public void schedule() {
    JobScheduler jobScheduler =
      (JobScheduler) getSystemService(Context.JOB_SCHEDULER_SERVICE);
    int id = jobId();

    // Check if the job is already scheduled and reschedule it in case the job
    // has changed.
    for (JobInfo ji : jobScheduler.getAllPendingJobs()) {
      if (ji.getId() == id) {
        jobScheduler.cancel(id);
      }
    }

    jobScheduler.schedule(new JobInfo.Builder(id,
          new ComponentName(this, this.getClass()))
        .setRequiredNetworkType(JobInfo.NETWORK_TYPE_UNMETERED)
        .setPersisted(true)
        .setRequiresCharging(true)
        .setRequiresDeviceIdle(true)
        .setPeriodic(intervalMillis())
        .build());
  }

  @Override
  public boolean onStartJob(final JobParameters params) {
    try {
      mt = new ModelType(domain(), modelType(), dataDir());
      mt.startTraining();
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
    return true;
  }

  @Override
  public boolean onStopJob(final JobParameters params) {
    if (mt != null) {
      try {
        mt.stopTraining();
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    }
    return true;
  }

  public int jobId() {
    return ("training:"+domain()+"/"+modelType()+"/"+dataDir()).hashCode();
  }

  // intervalMillis is the number of milliseconds that the training job will run
  // at most once per.
  public long intervalMillis() {
    // default to 6 hours.
    return 6 * 60 * 60 * 1000;
  }

  public abstract String domain();
  public abstract String modelType();
  public abstract String dataDir();
}
