# Apache Beam and Flink

This is just a demo of me tinkering with Beam and Flink

To run the demo flink job simply run
```
$ cd flink/
$ docker build .
$ docker-compose up
```
The output will look like so:

```
$ docker-compose up
Creating network "flink_default" with the default driver
Building web
[+] Building 125.1s (11/11) FINISHED                                                                                                                                                                                    
 => [internal] load build definition from Dockerfile                                                                                                                                                               0.0s
 => => transferring dockerfile: 339B                                                                                                                                                                               0.0s
 => [internal] load .dockerignore                                                                                                                                                                                  0.0s
 => => transferring context: 2B                                                                                                                                                                                    0.0s
 => [internal] load metadata for docker.io/library/python:3.8                                                                                                                                                      1.1s
 => [1/6] FROM docker.io/library/python:3.8@sha256:8b0fc420ec9b030c8c788ecd6dc9ad6b2f695dce183dc6a470f74c732e019a4a                                                                                                0.0s
 => [internal] load build context                                                                                                                                                                                  0.0s
 => => transferring context: 6.29kB                                                                                                                                                                                0.0s
 => CACHED [2/6] WORKDIR /code                                                                                                                                                                                     0.0s
 => [3/6] COPY ./* /code/                                                                                                                                                                                          0.1s
 => [4/6] RUN apt-get update &&     apt-get install -y openjdk-11-jre-headless &&     apt-get clean;                                                                                                              23.4s
 => [5/6] RUN export JAVA_HOME                                                                                                                                                                                     0.5s 
 => [6/6] RUN pip install -r /code/requirements.txt                                                                                                                                                               90.9s 
 => exporting to image                                                                                                                                                                                             8.9s 
 => => exporting layers                                                                                                                                                                                            8.9s 
 => => writing image sha256:7cd4929d84a49014419d6aed4f7c6ece83091a862ede77044f8811b80fe21c95                                                                                                                       0.0s 
 => => naming to docker.io/library/flink_web                                                                                                                                                                       0.0s 
Successfully built 7cd4929d84a49014419d6aed4f7c6ece83091a862ede77044f8811b80fe21c95                                                                                                                                     
WARNING: Image for service web was built because it did not already exist. To rebuild this image you must use `docker-compose build` or `docker-compose up --build`.
Creating flink_web_1 ... done
Attaching to flink_web_1
web_1  | WARNING: An illegal reflective access operation has occurred
web_1  | WARNING: Illegal reflective access by org.apache.flink.api.java.ClosureCleaner (file:/usr/local/lib/python3.8/site-packages/pyflink/lib/flink-dist_2.11-1.14.4.jar) to field java.lang.String.value
web_1  | WARNING: Please consider reporting this to the maintainers of org.apache.flink.api.java.ClosureCleaner
web_1  | WARNING: Use --illegal-access=warn to enable warnings of further illegal reflective access operations
web_1  | WARNING: All illegal access operations will be denied in a future release
web_1  | +I[To, 1]
web_1  | +I[be,, 1]
web_1  | +I[or, 1]
web_1  | +I[not, 1]
web_1  | +I[to, 1]
web_1  | +I[be,--that, 1]
web_1  | +I[is, 1]
web_1  | +I[the, 1]
web_1  | +I[question:--, 1]
web_1  | +I[Whether, 1]
web_1  | +I['tis, 1]
web_1  | +I[nobler, 1]
web_1  | +I[in, 1]
web_1  | -U[the, 1]
web_1  | +U[the, 2]
web_1  | +I[mind, 1]
web_1  | -U[to, 1]
web_1  | +U[to, 2]
web_1  | +I[suffer, 1]
web_1  | +I[The, 1]
web_1  | +I[slings, 1]
web_1  | +I[and, 1]
web_1  | +I[arrows, 1]
web_1  | +I[of, 1]
web_1  | +I[outrageous, 1]
web_1  | +I[fortune, 1]
web_1  | +I[Or, 1]
web_1  | -U[to, 2]
web_1  | +U[to, 3]
web_1  | +I[take, 1]
web_1  | +I[arms, 1]
web_1  | +I[against, 1]
web_1  | +I[a, 1]
web_1  | +I[sea, 1]
web_1  | -U[of, 1]
web_1  | +U[of, 2]
web_1  | +I[troubles,, 1]
web_1  | +I[And, 1]
web_1  | +I[by, 1]
web_1  | +I[opposing, 1]
web_1  | +I[end, 1]
web_1  | +I[them?--To, 1]
web_1  | +I[die,--to, 1]
web_1  | +I[sleep,--, 1]
web_1  | +I[No, 1]
web_1  | +I[more;, 1]
web_1  | -U[and, 1]
web_1  | +U[and, 2]
web_1  | -U[by, 1]
web_1  | +U[by, 2]
web_1  | -U[a, 1]
web_1  | +U[a, 2]
web_1  | +I[sleep, 1]
web_1  | -U[to, 3]
web_1  | +U[to, 4]
web_1  | +I[say, 1]
web_1  | +I[we, 1]
web_1  | -U[end, 1]
web_1  | +U[end, 2]
web_1  | -U[The, 1]
web_1  | +U[The, 2]
web_1  | +I[heartache,, 1]
web_1  | -U[and, 2]
web_1  | +U[and, 3]
web_1  | -U[the, 2]
web_1  | +U[the, 3]
web_1  | +I[thousand, 1]
web_1  | +I[natural, 1]
web_1  | +I[shocks, 1]
web_1  | +I[That, 1]
web_1  | +I[flesh, 1]
web_1  | -U[is, 1]
web_1  | +U[is, 2]
web_1  | +I[heir, 1]
web_1  | +I[to,--'tis, 1]
web_1  | -U[a, 2]
web_1  | +U[a, 3]
web_1  | +I[consummation, 1]
web_1  | +I[Devoutly, 1]
web_1  | -U[to, 4]
web_1  | +U[to, 5]
web_1  | +I[be, 1]
web_1  | +I[wish'd., 1]
web_1  | -U[To, 1]
web_1  | +U[To, 2]
web_1  | -U[die,--to, 1]
web_1  | +U[die,--to, 2]
web_1  | +I[sleep;--, 1]
web_1  | -U[To, 2]
web_1  | +U[To, 3]
web_1  | +I[sleep!, 1]
web_1  | +I[perchance, 1]
web_1  | -U[to, 5]
web_1  | +U[to, 6]
web_1  | +I[dream:--ay,, 1]
web_1  | +I[there's, 1]
web_1  | -U[the, 3]
web_1  | +U[the, 4]
web_1  | +I[rub;, 1]
web_1  | +I[For, 1]
web_1  | -U[in, 1]
web_1  | +U[in, 2]
web_1  | +I[that, 1]
web_1  | -U[sleep, 1]
web_1  | +U[sleep, 2]
web_1  | -U[of, 2]
web_1  | +U[of, 3]
web_1  | +I[death, 1]
web_1  | +I[what, 1]
web_1  | +I[dreams, 1]
web_1  | +I[may, 1]
web_1  | +I[come,, 1]
web_1  | +I[When, 1]
web_1  | -U[we, 1]
web_1  | +U[we, 2]
web_1  | +I[have, 1]
web_1  | +I[shuffled, 1]
web_1  | +I[off, 1]
web_1  | +I[this, 1]
web_1  | +I[mortal, 1]
web_1  | +I[coil,, 1]
web_1  | +I[Must, 1]
web_1  | +I[give, 1]
web_1  | +I[us, 1]
web_1  | +I[pause:, 1]
web_1  | -U[there's, 1]
web_1  | +U[there's, 2]
web_1  | -U[the, 4]
web_1  | +U[the, 5]
web_1  | +I[respect, 1]
web_1  | -U[That, 1]
web_1  | +U[That, 2]
web_1  | +I[makes, 1]
web_1  | +I[calamity, 1]
web_1  | -U[of, 3]
web_1  | +U[of, 4]
web_1  | +I[so, 1]
web_1  | +I[long, 1]
web_1  | +I[life;, 1]
web_1  | -U[For, 1]
web_1  | +U[For, 2]
web_1  | +I[who, 1]
web_1  | +I[would, 1]
web_1  | +I[bear, 1]
web_1  | -U[the, 5]
web_1  | +U[the, 6]
web_1  | +I[whips, 1]
web_1  | -U[and, 3]
web_1  | +U[and, 4]
web_1  | +I[scorns, 1]
web_1  | -U[of, 4]
web_1  | +U[of, 5]
web_1  | +I[time,, 1]
web_1  | -U[The, 2]
web_1  | +U[The, 3]
web_1  | +I[oppressor's, 1]
web_1  | +I[wrong,, 1]
web_1  | -U[the, 6]
web_1  | +U[the, 7]
web_1  | +I[proud, 1]
web_1  | +I[man's, 1]
web_1  | +I[contumely,, 1]
web_1  | -U[The, 3]
web_1  | +U[The, 4]
web_1  | +I[pangs, 1]
web_1  | -U[of, 5]
web_1  | +U[of, 6]
web_1  | +I[despis'd, 1]
web_1  | +I[love,, 1]
web_1  | -U[the, 7]
web_1  | +U[the, 8]
web_1  | +I[law's, 1]
web_1  | +I[delay,, 1]
web_1  | -U[The, 4]
web_1  | +U[The, 5]
web_1  | +I[insolence, 1]
web_1  | -U[of, 6]
web_1  | +U[of, 7]
web_1  | +I[office,, 1]
web_1  | -U[and, 4]
web_1  | +U[and, 5]
web_1  | -U[the, 8]
web_1  | +U[the, 9]
web_1  | +I[spurns, 1]
web_1  | -U[That, 2]
web_1  | +U[That, 3]
web_1  | +I[patient, 1]
web_1  | +I[merit, 1]
web_1  | -U[of, 7]
web_1  | +U[of, 8]
web_1  | -U[the, 9]
web_1  | +U[the, 10]
web_1  | +I[unworthy, 1]
web_1  | +I[takes,, 1]
web_1  | -U[When, 1]
web_1  | +U[When, 2]
web_1  | +I[he, 1]
web_1  | +I[himself, 1]
web_1  | +I[might, 1]
web_1  | +I[his, 1]
web_1  | +I[quietus, 1]
web_1  | +I[make, 1]
web_1  | +I[With, 1]
web_1  | -U[a, 3]
web_1  | +U[a, 4]
web_1  | +I[bare, 1]
web_1  | +I[bodkin?, 1]
web_1  | -U[who, 1]
web_1  | +U[who, 2]
web_1  | -U[would, 1]
web_1  | +U[would, 2]
web_1  | +I[these, 1]
web_1  | +I[fardels, 1]
web_1  | +I[bear,, 1]
web_1  | -U[To, 3]
web_1  | +U[To, 4]
web_1  | +I[grunt, 1]
web_1  | -U[and, 5]
web_1  | +U[and, 6]
web_1  | +I[sweat, 1]
web_1  | +I[under, 1]
web_1  | -U[a, 4]
web_1  | +U[a, 5]
web_1  | +I[weary, 1]
web_1  | +I[life,, 1]
web_1  | +I[But, 1]
web_1  | -U[that, 1]
web_1  | +U[that, 2]
web_1  | -U[the, 10]
web_1  | +U[the, 11]
web_1  | +I[dread, 1]
web_1  | -U[of, 8]
web_1  | +U[of, 9]
web_1  | +I[something, 1]
web_1  | +I[after, 1]
web_1  | +I[death,--, 1]
web_1  | -U[The, 5]
web_1  | +U[The, 6]
web_1  | +I[undiscover'd, 1]
web_1  | +I[country,, 1]
web_1  | +I[from, 1]
web_1  | +I[whose, 1]
web_1  | +I[bourn, 1]
web_1  | -U[No, 1]
web_1  | +U[No, 2]
web_1  | +I[traveller, 1]
web_1  | +I[returns,--puzzles, 1]
web_1  | -U[the, 11]
web_1  | +U[the, 12]
web_1  | +I[will,, 1]
web_1  | -U[And, 1]
web_1  | +U[And, 2]
web_1  | -U[makes, 1]
web_1  | +U[makes, 2]
web_1  | -U[us, 1]
web_1  | +U[us, 2]
web_1  | +I[rather, 1]
web_1  | -U[bear, 1]
web_1  | +U[bear, 2]
web_1  | +I[those, 1]
web_1  | +I[ills, 1]
web_1  | -U[we, 2]
web_1  | +U[we, 3]
web_1  | -U[have, 1]
web_1  | +U[have, 2]
web_1  | +I[Than, 1]
web_1  | +I[fly, 1]
web_1  | -U[to, 6]
web_1  | +U[to, 7]
web_1  | +I[others, 1]
web_1  | -U[that, 2]
web_1  | +U[that, 3]
web_1  | -U[we, 3]
web_1  | +U[we, 4]
web_1  | +I[know, 1]
web_1  | -U[not, 1]
web_1  | +U[not, 2]
web_1  | +I[of?, 1]
web_1  | +I[Thus, 1]
web_1  | +I[conscience, 1]
web_1  | +I[does, 1]
web_1  | -U[make, 1]
web_1  | +U[make, 2]
web_1  | +I[cowards, 1]
web_1  | -U[of, 9]
web_1  | +U[of, 10]
web_1  | -U[us, 2]
web_1  | +U[us, 3]
web_1  | +I[all;, 1]
web_1  | -U[And, 2]
web_1  | +U[And, 3]
web_1  | +I[thus, 1]
web_1  | -U[the, 12]
web_1  | +U[the, 13]
web_1  | +I[native, 1]
web_1  | +I[hue, 1]
web_1  | -U[of, 10]
web_1  | +U[of, 11]
web_1  | +I[resolution, 1]
web_1  | +I[Is, 1]
web_1  | +I[sicklied, 1]
web_1  | +I[o'er, 1]
web_1  | +I[with, 1]
web_1  | -U[the, 13]
web_1  | +U[the, 14]
web_1  | +I[pale, 1]
web_1  | +I[cast, 1]
web_1  | -U[of, 11]
web_1  | +U[of, 12]
web_1  | +I[thought;, 1]
web_1  | -U[And, 3]
web_1  | +U[And, 4]
web_1  | +I[enterprises, 1]
web_1  | -U[of, 12]
web_1  | +U[of, 13]
web_1  | +I[great, 1]
web_1  | +I[pith, 1]
web_1  | -U[and, 6]
web_1  | +U[and, 7]
web_1  | +I[moment,, 1]
web_1  | -U[With, 1]
web_1  | +U[With, 2]
web_1  | -U[this, 1]
web_1  | +U[this, 2]
web_1  | +I[regard,, 1]
web_1  | +I[their, 1]
web_1  | +I[currents, 1]
web_1  | +I[turn, 1]
web_1  | +I[awry,, 1]
web_1  | -U[And, 4]
web_1  | +U[And, 5]
web_1  | +I[lose, 1]
web_1  | -U[the, 14]
web_1  | +U[the, 15]
web_1  | +I[name, 1]
web_1  | -U[of, 13]
web_1  | +U[of, 14]
web_1  | +I[action.--Soft, 1]
web_1  | +I[you, 1]
web_1  | +I[now!, 1]
web_1  | -U[The, 6]
web_1  | +U[The, 7]
web_1  | +I[fair, 1]
web_1  | +I[Ophelia!--Nymph,, 1]
web_1  | -U[in, 2]
web_1  | +U[in, 3]
web_1  | +I[thy, 1]
web_1  | +I[orisons, 1]
web_1  | +I[Be, 1]
web_1  | +I[all, 1]
web_1  | +I[my, 1]
web_1  | +I[sins, 1]
web_1  | +I[remember'd., 1]
web_1  | Executing word_count example with default input data set.
web_1  | Use --input to specify file input.
web_1  | Printing result to stdout. Use --output to specify output path.
flink_web_1 exited with code 0
```