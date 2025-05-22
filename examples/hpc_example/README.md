

# Running the scripts on BluePebble

To run the script on Bristol BluePebble:

Compile singularity on your machine:

First cd to the right folder:

``` cd path_to_Abil/examples/singularity/  ```

If using apptainer (recommended):

``` sudo apptainer build abil.sif Singularity.sif  ```

If using singularity:

``` sudo singularity build abil.sif Singularity.sif  ```

Note: apptainer is often easier to install than singularity and is backwards compatible with legacy Singularity installs:

https://apptainer.org/docs/admin/1.2/installation.html

## Transfer Abil to the cluster.

To upload only relevant files with scp there is a `to-bp.sh` bash script.

To run the script, cd to the Abil directory:

``` cd path_to_Abil ```

Set the bash script executable permission by running chmod command in Linux:

``` sudo chmod +x to-bp.sh ```

Edit the file to your own username.

Then execute it:

``` sudo ./to-bp.sh ``` 

The script will ask for your cluster login information and then transfer the files.



Connect to BluePebble:

``` ssh my_username@bp1-login.acrc.bris.ac.uK ``` 


Change directory to abil:

``` cd /user/work/my_username/abil ``` 


Change directory to folder containing you bash scripts:

``` cd cluster_example ``` 


Submit cluster job:

``` sbatch tune_RF.sh ``` 


Check if the job is running:

``` sacct ``` 


Delete error output files:

``` rm *.out ```
