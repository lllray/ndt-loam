#include <ndt_mcl/ParticleFilter3D.h>
/*
 * Implementation for the 6dof pose particle filtering
 */ 

/**
 * Initializes the filter using normally distributed random variables with given means (m-values) and standard deviations (v-values)
 */ 
void ParticleFilter3D::initializeNormalRandom(unsigned int NumParticles, double mx, double my, double mz, double mroll, double mpitch, double myaw,
																									 double vx, double vy, double vz, double vroll, double vpitch, double vyaw)
{
	for(unsigned int i=0;i<NumParticles;i++){
		
		double x = mx + myrand.normalRandom() * vx;
		double y = my + myrand.normalRandom() * vy;
		double z = mz + myrand.normalRandom() * vz;
		
		double roll 	= mroll + myrand.normalRandom() * vroll;
		double pitch = mpitch + myrand.normalRandom() * vpitch;
		double yaw 	= myaw + myrand.normalRandom() * vyaw;
		
		PoseParticle P(x,y,z,roll,pitch,yaw);
		P.lik = 1.0;
		P.p = 1.0 / (double) NumParticles;
		pcloud.push_back(P);
	} 
}

/**
* Performs the Sample Importance Resampling (SIR) algorithm for the distribution
* The algorithm chooses the best particles (with respect to the probability) and 
* resamples these. 
* 
* You should have updated the likelihoods and normalized the distribution before running this
* Also, it might be smart not to run this in every iteration, since the distribution looses accuracy
* due to the "discretation"
**/
void ParticleFilter3D::SIRUpdate(){
		std::vector<PoseParticle> tmp;
		tmp.resize(pcloud.size());
		double U=0,Q=0;
		int i=0,j=0,k=0;
		
		int NumOfParticles = pcloud.size();
		U = myrand.uniformRandom() / (double) NumOfParticles;
		//fprintf(stderr,"SIRUpdate()::U=%.6f\n",U);
	
		
		while(U < 1.0){
		
				if(Q>U){ ///<-- Replicate the particle
						U += 1.0/(double)NumOfParticles;
						
						/// Check for index error 
						if(k>=NumOfParticles || i>=NumOfParticles){
								fprintf(stderr,"SIR error i=%d k=%d N=%d",i,k,NumOfParticles);
								break; ///Leave the loop 
						}
						tmp[i]=pcloud[k];
						tmp[i].p = 1.0 / (double)NumOfParticles;
						i++;
				}
				else{ ///Moving on
						j++;
						k=j;
						
						if(j>=NumOfParticles){ ///Index exceeded
								//fprintf(stderr,"SIR error(2) i=%d k=%d N=%d",i,k,NumOfParticles);
								break; ///Leave the loop     
						}
						Q += pcloud[j].p; ///< add the weight to cumulative sum
				}
		}//While
		
		if(i<(NumOfParticles-1)) fprintf(stderr,"SIR error(3) i=%d k=%d N=%d\n",i,k,NumOfParticles);
		while(i<NumOfParticles){ ///Make sure that the vector is filled
			  if(k>=NumOfParticles) k=NumOfParticles-1;
				tmp[i]=pcloud[k];
				tmp[i].p = 1.0 / NumOfParticles;
				i++;
		}
		
		pcloud = tmp;
}

/**
 * Performs the normalization step
 * i.e. according to updated likelyhoods the probability of each 
 * particle is calculated and the whole distribution gets 
 * probablity of 1
 */
void ParticleFilter3D::normalize(){
		int i;
		double summ=0;
		
		for(unsigned i=0;i<pcloud.size();i++){
				pcloud[i].p *= pcloud[i].lik; 
				summ+=pcloud[i].p;
		}
		if(summ > 0){
			for(i=0;i<pcloud.size();i++){
				pcloud[i].p = pcloud[i].p/summ;
			}
		}else{
			for(i=0;i<pcloud.size();i++){
				pcloud[i].p = 1.0/(double)pcloud.size();
			}
		}
}

void ParticleFilter3D::predict(Eigen::Affine3d Tmotion, double vx, double vy, double vz, double vroll, double vpitch, double vyaw,const Eigen::Affine3d offset){
	Eigen::Vector3d tr = Tmotion.translation();
	Eigen::Vector3d rot = Tmotion.rotation().eulerAngles(0,1,2);
  bool transform_cloud=false;
  if(offset.translation().norm()>0.0001 && offset.rotation().eulerAngles(0,1,2).norm()>0.0001)
    transform_cloud=true;

  /* #pragma omp parallel num_threads(8)
  {
    #pragma omp for*/
	for(unsigned int i=0;i<pcloud.size();i++){
		double x = tr[0] + myrand.normalRandom() * vx;
		double y = tr[1] + myrand.normalRandom() * vy;
		double z = tr[2] + myrand.normalRandom() * vz;
		
		double roll 	= rot[0] + myrand.normalRandom() * vroll;
		double pitch = rot[1] + myrand.normalRandom() * vpitch;
		double yaw 	= rot[2] + myrand.normalRandom() * vyaw;
		
		pcloud[i].T = pcloud[i].T *(xyzrpy2affine(x,y,z,roll,pitch,yaw));
    if(transform_cloud)
      pcloud[i].T = offset*pcloud[i].T ;
  ///}
}
}

Eigen::Affine3d ParticleFilter3D::getMean(){
	double mx=0, my=0,mz=0;
	//Eigen::Quaternion<double> qm;
	double roll_x = 0, roll_y=0;
	double pitch_x = 0, pitch_y=0;
	double yaw_x = 0, yaw_y=0;
	
        Eigen::Matrix3d sumRot = Eigen::Matrix3d::Zero();
        
	for(unsigned int i=0;i<pcloud.size();i++){		
	    //Eigen::Quaternion<double> q(pcloud[i].T.rotation());
	    //qm=qm+pcloud[i].p * q;
	    Eigen::Vector3d tr = pcloud[i].T.translation();
	    mx += pcloud[i].p * tr[0];
	    my += pcloud[i].p * tr[1];
	    mz += pcloud[i].p * tr[2];

	    //Get as euler
	    Eigen::Vector3d rot = pcloud[i].T.rotation().eulerAngles(0,1,2);
	    roll_x+=pcloud[i].p*cos(rot[0]); 
	    roll_y+=pcloud[i].p*sin(rot[0]);

	    pitch_x+=pcloud[i].p*cos(rot[1]); 
	    pitch_y+=pcloud[i].p*sin(rot[1]);

	    yaw_x+=pcloud[i].p*cos(rot[2]); 
	    yaw_y+=pcloud[i].p*sin(rot[2]);

            sumRot += pcloud[i].p*pcloud[i].T.rotation();
	}

	Eigen::JacobiSVD<Eigen::Matrix3d> svd(sumRot,  Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d averageRot = svd.matrixU() * svd.matrixV().transpose();

        Eigen::Affine3d ret;
        ret.translation() = Eigen::Vector3d(mx,my,mz);
        ret.linear() = averageRot;
        return ret; //Eigen::Translation3d v(mx,my,mz)*Eigen::Rotation3d(averageRot);

        //	return xyzrpy2affine(mx,my,mz, atan2(roll_y,roll_x), atan2(pitch_y,pitch_x), atan2(yaw_y,yaw_x));
	
	
	//qm.normalize();
	//Eigen::Matrix3d m;
	//m = qm.toRotationMatrix();
	//Eigen::Translation3d v(mx,my,mz);
}

#if 0 // no one cares
Eigen::Matrix<double,7,7> ParticleFilter3D::getCov() {

    Eigen::Affine3d mean = this->getMean();
    Eigen::Quaterniond mean_r = mean.rotation(), qt;
    Eigen::MatrixXd mt = Eigen::MatrixXd(7,pcloud.size());
    Eigen::Vector3d t1;
    Eigen::Vector4d t2;

    Eigen::Matrix<double,7,7> cov;
    cov.setIdentity();

    for(unsigned int i=0;i<pcloud.size();i++){		
	    //Eigen::Quaternion<double> q(pcloud[i].T.rotation());
	    //qm=qm+pcloud[i].p * q;
	    t1 = pcloud[i].T.translation() - mean.translation();
	    qt = pcloud[i].T.rotation();
	    t2 = qt - mean_r;
	    mt(0,i) = t1(0);
	    mt(1,i) = t1(1);
	    mt(2,i) = t1(2);
	    
	    mt(1,i) = t2(0);
	    mt(2,i) = t2(1);
	    mt(3,i) = t2(2);
	    mt(4,i) = t2(3);
	}
    cov = mt*mt.transpose() / pcloud.size();
}
#endif


