#include <Eigen/Eigenvalues>
#include "ndt_mcl/particle_filter.hpp"
#include "ros/ros.h"

bool myfunction (double i,double j) { return (i>j); }

perception_oru::particle_filter::particle_filter(perception_oru::NDTMap *ndtMap_, int particleCount_ /*, init_type initializationType_*/, bool be2D_, bool forceSIR_, double varLimit_, int sirCount_){
	be2D = be2D_;
	ndtMap = ndtMap_;
	// initializationType=initializationType_;
	particleCount = particleCount_;
	forceSIR = forceSIR_;
	varLimit = varLimit_;
	sirCount = sirCount_;
	weights.resize(particleCount_,0);
}

void perception_oru::particle_filter::Reset(){
	tmp.clear();
	delete ndt_ISSMap;
	particleCloud.clear();
}

void perception_oru::particle_filter::InitializeNormal(double x, double y, double th, double var){

	if(particleCloud.size() > 0){
		particleCloud.clear();
		tmp.clear();
	}
	m_pose<<x,y,th;
	tmp.resize(particleCount);
	std::default_random_engine generator;

	for(int parNo = 0; parNo < particleCount; parNo++){
		std::normal_distribution<double> distribution_x(x, var);
		std::normal_distribution<double> distribution_y(y, var);
		std::normal_distribution<double> distribution_t(th, th * var);
		particleCloud.emplace_back(0.0, 0.0, distribution_t(generator), distribution_x(generator), distribution_y(generator), 0.0);
	}
}

void perception_oru::particle_filter::InitializeNormal(double x, double y, double var){

	if(particleCloud.size() > 0){
		particleCloud.clear();
		tmp.clear();
	}
		m_pose<<x,y,0;
	tmp.resize(particleCount);
	std::default_random_engine generator;
	for(int parNo = 0; parNo < particleCount; parNo++){
		std::normal_distribution<double> distribution_x(x, var);
		std::normal_distribution<double> distribution_y(y, var);
		std::uniform_real_distribution<double> distribution_t(0, 2 * 3.1415);
		particleCloud.emplace_back(0.0, 0.0, distribution_t(generator), distribution_x(generator), distribution_y(generator), 0.0);
	}
}


Eigen::Vector3d perception_oru::particle_filter::GetMeanPose2D(){
	double sumX = 0, sumY = 0;
	Eigen::Vector3d pos;
	double sumW = 0;
	double ax = 0, ay = 0;

	for(int i = 0; i < particleCloud.size(); i++){
		double x, y, z, r, p, t;
		particleCloud[i].GetXYZ(x, y, z);
		particleCloud[i].GetRPY(r, p, t);
		sumX += particleCloud[i].GetProbability() * x;
		sumY += particleCloud[i].GetProbability() * y;
		ax += particleCloud[i].GetProbability() * cos(t);
		ay += particleCloud[i].GetProbability() * sin(t);
		sumW += particleCloud[i].GetProbability();
	}
	pos << sumX, sumY, atan2(ay, ax);
	return pos;
}


void perception_oru::particle_filter::GetPoseMeanAndVariance2D(Eigen::Vector3d &mean, Eigen::Matrix3d &cov){
	double sumX = 0, sumY = 0;
	double sumW = 0;
	double ax = 0, ay = 0;

	for(int i = 0; i < particleCloud.size(); i++){
		double x, y, z, r, p, t;
		particleCloud[i].GetXYZ(x, y, z);
		particleCloud[i].GetRPY(r, p, t);
		sumX += particleCloud[i].GetProbability() * x;
		sumY += particleCloud[i].GetProbability() * y;
		ax += particleCloud[i].GetProbability() * cos(t);
		ay += particleCloud[i].GetProbability() * sin(t);
		sumW += particleCloud[i].GetProbability();
	}
	mean << sumX, sumY, atan2(ay, ax);

	double xx = 0, yy = 0, xy = 0, aax = 0, aay = 0;
	double cax = cos(atan2(ay, ax));
	double say = sin(atan2(ay, ax));
	double w2 = 0;


	for(int i = 0; i < particleCloud.size(); i++){
		double x, y, z, r, p, t;
		particleCloud[i].GetXYZ(x, y, z);
		particleCloud[i].GetRPY(r, p, t);
		xx += particleCloud[i].GetProbability() * (x - sumX) * (x - sumX);
		yy += particleCloud[i].GetProbability() * (y - sumY) * (y - sumY);
		xy += particleCloud[i].GetProbability() * (x - sumX) * (y - sumY);
		aax += particleCloud[i].GetProbability() * (cos(t) - cax) * (cos(t) - cax);
		aay += particleCloud[i].GetProbability() * (sin(t) - say) * (sin(t) - say);
		w2 += particleCloud[i].GetProbability() * particleCloud[i].GetProbability();
	}

	if(w2 == 1.0){
		fprintf(stderr, "CParticleFilter::getDistributionVariances -- w2=%lf Should not happen!\n", w2);
		w2 = 0.99;
	}
	double wc = 1.0 / (1.0 - w2);
	cov << wc * xx, wc * xy, 0,
	wc * xy, wc * yy, 0,
	0, 0, atan2(wc * aay, wc * aax);
}

void perception_oru::particle_filter::InitializeUniformMap(){
	int pCount_ = particleCount;

	std::vector<perception_oru::NDTCell*> allCells = ndtMap->getAllInitializedCells();
	std::vector<perception_oru::NDTCell*> cells;
	for(int cInd = 0; cInd < allCells.size(); cInd++)
		if(allCells[cInd]->getOccupancy() < 0.0)
			cells.push_back(allCells[cInd]);

	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution_c(0, cells.size() - 1);
	tmp.resize(pCount_);
	for(int parNo = 0; parNo < pCount_; parNo++){
		int cellId = distribution_c(generator);
		double cx, cy, cz, sx, sy, sz;
		if(be2D){
			cells[cellId]->getCenter(cx, cy, cz);
			cells[cellId]->getDimensions(sx, sy, sz);
			std::uniform_real_distribution<double> distribution_x(cx - sx / 2.0, cx + sx / 2.0);
			std::uniform_real_distribution<double> distribution_y(cy - sy / 2.0, cy + sy / 2.0);
			std::uniform_real_distribution<double> distribution_t(0.0, 2 * M_PI);
			particleCloud.emplace_back(0.0, 0.0, distribution_t(generator), distribution_x(generator), distribution_y(generator), 0.0);
		}
		//here will go 3d distibution
	}
}

void perception_oru::particle_filter::InitializeFilter(){
	// switch(initializationType){
	//case uniform_map:
	InitializeUniformMap();
	// break;
	//  default:3
	//}
}

void perception_oru::particle_filter::UpdateAndPredict(Eigen::Affine3d tMotion, perception_oru::NDTMap ndtLocalMap_){
	Eigen::Vector3d tr = tMotion.translation();
	Eigen::Vector3d rot = tMotion.rotation().eulerAngles(0, 1, 2);

	if(tr[0] != 0.0 && tr[1] != 0.0 && rot[2] != 0){
		Predict2D(tr[0], tr[1], rot[2], tr[0] * 0.03 + 0.005, tr[1] * 0.03 + 0.005, rot[2] * 0.08 + 0.01);
				#pragma omp parallel for
		for(int i = 0; i < particleCloud.size(); i++){
			Eigen::Affine3d T = particleCloud[i].GetAsAffine();
			std::vector<perception_oru::NDTCell*> ndts;
			ndts = ndtLocalMap_.pseudoTransformNDT(T);
			double score = 1;
			if(ndts.size() == 0) fprintf(stderr, "ERROR no gaussians in measurement!!!\n");
			for(int n = 0; n < ndts.size(); n++){
				Eigen::Vector3d m = ndts[n]->getMean();
				perception_oru::NDTCell *cell;
				pcl::PointXYZ p;
				p.x = m[0]; p.y = m[1]; p.z = m[2];
				if(ndtMap->getCellForPoint(p, cell)){
					if(cell == NULL) continue;
					if(cell->hasGaussian_){
						Eigen::Matrix3d covCombined = cell->getCov() + ndts[n]->getCov();
						Eigen::Matrix3d icov;
						bool exists;
						double det = 0;
						covCombined.computeInverseAndDetWithCheck(icov, det, exists);
						if(!exists) continue;
						double l = (cell->getMean() - m).dot(icov * (cell->getMean() - m));
						if(l * 0 != 0) continue;
						score += 0.1 + 0.9 * exp(-0.05 * l / 2.0);
					}else {
					}
				}
			}
			particleCloud[i].SetLikelihood(score);
			for(unsigned int j = 0; j < ndts.size(); j++)
				delete ndts[j];
		}
		Normalize();
		if(forceSIR)
			SIRUpdate();
		else{
			double varP = 0;
			for(int i = 0; i < particleCloud.size(); i++)
				varP += (particleCloud[i].GetProbability() - 1.0 / double(particleCloud.size()))
				        * (particleCloud[i].GetProbability() - 1.0 / double(particleCloud.size()));
			varP /= double(particleCloud.size());
			varP = sqrt(varP);
			if(varP > varLimit || sinceSIR > sirCount){
				//fprintf(stderr,"-SIR- ");
				sinceSIR = 0;
				SIRUpdate();
			}else
				sinceSIR++;
		}
	}
}
void perception_oru::particle_filter::UpdateAndPredict(Eigen::Affine3d tMotion, perception_oru::NDTMap* ndtLocalMap_){
	Eigen::Vector3d tr = tMotion.translation();
	Eigen::Vector3d rot = tMotion.rotation().eulerAngles(0, 1, 2);

	if(tr[0] != 0.0 && tr[1] != 0.0 && rot[2] != 0){
		Predict2D(tr[0], tr[1], rot[2], tr[0] * 0.1 + 0.005, tr[1] * 0.1 + 0.005, rot[2] * 0.1 + 0.001);
				#pragma omp parallel for
		for(int i = 0; i < particleCloud.size(); i++){
			Eigen::Affine3d T = particleCloud[i].GetAsAffine();
			std::vector<perception_oru::NDTCell*> ndts;
			ndts = ndtLocalMap_->pseudoTransformNDT(T);
			double score = 1;
			if(ndts.size() == 0) fprintf(stderr, "ERROR no gaussians in measurement!!!\n");
			for(int n = 0; n < ndts.size(); n++){
				Eigen::Vector3d m = ndts[n]->getMean();
				perception_oru::NDTCell *cell;
				//pcl::PointXYZ p;
				//p.x = m[0];p.y=m[1];p.z=m[2];
				pcl::PointXYZ p(m[0], m[1], m[2]);
				if(ndtMap->getCellForPoint(p, cell)){
					if(cell == NULL) continue;
					if(cell->hasGaussian_){
						Eigen::Matrix3d covCombined = cell->getCov() + ndts[n]->getCov();
						Eigen::Matrix3d icov;
						bool exists;
						double det = 0;
						covCombined.computeInverseAndDetWithCheck(icov, det, exists);
						if(!exists) continue;
						double l = (cell->getMean() - m).dot(icov * (cell->getMean() - m));
						if(l * 0 != 0) continue;
						score += 0.1 + 0.9 * exp(-0.05 * l / 2.0);
					}else {
					}
				}
			}
			particleCloud[i].SetLikelihood(score);
			for(unsigned int j = 0; j < ndts.size(); j++)
				delete ndts[j];
		}
		Normalize();
		if(forceSIR)
			SIRUpdate();
		else{
			double varP = 0;
			for(int i = 0; i < particleCloud.size(); i++)
				varP += (particleCloud[i].GetProbability() - 1.0 / double(particleCloud.size()))
				        * (particleCloud[i].GetProbability() - 1.0 / double(particleCloud.size()));
			varP /= double(particleCloud.size());
			varP = sqrt(varP);
			if(varP > varLimit || sinceSIR > sirCount){
				//fprintf(stderr,"-SIR- ");
				sinceSIR = 0;
				SIRUpdate();
			}else
				sinceSIR++;
		}
	}
}


void perception_oru::particle_filter::UpdateAndPredictEff(Eigen::Affine3d tMotion, perception_oru::NDTMap* ndtLocalMap_, double subsample_level, double z_cut){ //you may add z cut here if necessarry
	if(subsample_level < 0 || subsample_level > 1) subsample_level = 1;

	Eigen::Vector3d tr = tMotion.translation();
	Eigen::Vector3d rot = tMotion.rotation().eulerAngles(0, 1, 2);
	if(tr[0] != 0.0 && tr[1] != 0.0 && rot[2] != 0){
	  int all_cells;
	  int subsample_cells;
	  int used_cells=0;
	  Predict2D(tr[0], tr[1], rot[2], fabs(tr[0]) * 0.50 + 0.001, fabs(tr[1]) * 0.70 + 0.001, fabs(rot[2]) * 0.15 + 0.001); //this line!!!!!!!!!!!!!!!!!!!!!!!!!
	  //Predict2D(tr[0], tr[1], rot[2], fabs(tr[0]) * 0.25 + 0.001, fabs(tr[1]) * 0.25 + 0.001, fabs(rot[2]) * 0.15 + 0.001); //this line!!!!!!!!!!!!!!!!!!!!!!!!!
		std::vector<perception_oru::NDTCell*> ndts0 = ndtLocalMap_->getAllCells();
		std::vector<perception_oru::NDTCell*> ndts;
		if(subsample_level != 1){
			srand(time(NULL));
			for(int i = 0; i < ndts0.size(); ++i){
				double p = ((double)rand()) / RAND_MAX;

				if(p < subsample_level && ndts0[i]->getMean()[2]>z_cut && ndts0[i]->getClass()!=perception_oru::NDTCell::HORIZONTAL)
					ndts.push_back(ndts0[i]);
				else
					delete ndts0[i];
			}
		} else
			ndts = ndts0;
//#pragma omp parallel for
		for(int i = 0; i < particleCloud.size(); i++){
			Eigen::Affine3d T = particleCloud[i].GetAsAffine();
			// std::vector<lslgeneric::NDTCell*> ndts;
			// ndts = ndtLocalMap_->pseudoTransformNDT(T);
			double score = 1;
			if(ndts.size() == 0) fprintf(stderr, "ERROR no gaussians in measurement!!!\n");
			for(int n = 0; n < ndts.size(); n++){
				Eigen::Vector3d m = T * ndts[n]->getMean();
				//if(m[2] < zfilt_min) continue;
				perception_oru::NDTCell *cell;
				pcl::PointXYZ p;
				p.x = m[0]; p.y = m[1]; p.z = m[2];
				if(ndtMap->getCellAtPoint(p, cell)){
					if(cell == NULL) continue;
					if(cell->getClass()==perception_oru::NDTCell::HORIZONTAL) continue;
					if(cell->hasGaussian_){
					  	Eigen::Matrix3d map_cov =
						  cell->getCov();
						//cell->blurCov( 10 );

						//Eigen::Matrix3d covCombined = cell->getCov() + T.rotation() * ndts[n]->getCov() * T.rotation().transpose();
						Eigen::Matrix3d covCombined = map_cov + T.rotation() * ndts[n]->getCov() * T.rotation().transpose();
						Eigen::Matrix3d icov;
						bool exists;
						double det = 0;
						covCombined.computeInverseAndDetWithCheck(icov, det, exists);
						if(!exists) continue;
						double l = (cell->getMean() - m).dot(icov * (cell->getMean() - m));
						if(l * 0 != 0) continue;
						score += 0.1 + 0.9 * exp(-0.005 * l / 2.0);
					}else{

					}
				}
			}
			particleCloud[i].SetLikelihood(score);
		}
		for(unsigned int j = 0; j < ndts.size(); j++)
			delete ndts[j];

		Normalize();
		if(forceSIR)
			SIRUpdate();
		else{
			double varP = 0;
			for(int i = 0; i < particleCloud.size(); i++)
				varP += (particleCloud[i].GetProbability() - 1.0 / double(particleCloud.size()))
				        * (particleCloud[i].GetProbability() - 1.0 / double(particleCloud.size()));
			varP /= double(particleCloud.size());
			varP = sqrt(varP);
			if(varP > varLimit || sinceSIR > sirCount){
				//fprintf(stderr,"-SIR- ");
				sinceSIR = 0;
				SIRUpdate();
			}else
				sinceSIR++;
		}
	}
}



void perception_oru::particle_filter::UpdateAndPredictEffRe(Eigen::Affine3d tMotion, perception_oru::NDTMap* ndtLocalMap_, double subsample_level, double z_cut, double x_var, double y_var, double th_var, double r_x_var, double r_y_var, double r_th_var, int tres){
	if(subsample_level < 0 || subsample_level > 1) subsample_level = 1;
	double tre=weights[tres];
	Eigen::Vector3d tr = tMotion.translation();
	Eigen::Vector3d rot = tMotion.rotation().eulerAngles(0, 1, 2);
	if(tr[0] != 0.0 || tr[1] != 0.0 || rot[2] != 0){
	  int all_cells;
	  int subsample_cells;
	  int used_cells=0;
	  Predict2D(tr[0], tr[1], rot[2], fabs(tr[0]) * x_var, fabs(tr[1]) * y_var, fabs(rot[2]) * th_var); //this line!!!!!!!!!!!!!!!!!!!!!!!!!
		std::vector<perception_oru::NDTCell*> ndts0 = ndtLocalMap_->getAllCells();
		std::vector<perception_oru::NDTCell*> ndts;
		if(subsample_level != 1){
			srand(time(NULL));
			for(int i = 0; i < ndts0.size(); ++i){
				double p = ((double)rand()) / RAND_MAX;

				if(p < subsample_level && ndts0[i]->getMean()[2]>z_cut && ndts0[i]->getClass()!=perception_oru::NDTCell::HORIZONTAL)
					ndts.push_back(ndts0[i]);
				else
					delete ndts0[i];
			}
		} else
			ndts = ndts0;

//#pragma omp parallel for
		for(int i = 0; i < particleCloud.size(); i++){
			Eigen::Affine3d T;
			if(particleCloud[i].GetProbability()>tre){
				T = particleCloud[i].GetAsAffine();
			}
			else{
				std::default_random_engine generator;
				std::normal_distribution<double> distribution_x(m_pose[0], r_x_var);
				std::normal_distribution<double> distribution_y(m_pose[1], r_y_var);
				std::normal_distribution<double> distribution_t(m_pose[2], r_th_var);
				particleCloud[i].Set(0.0, 0.0, distribution_t(generator), distribution_x(generator), distribution_y(generator), 0.0);

				T = particleCloud[i].GetAsAffine();
			}
				// std::vector<lslgeneric::NDTCell*> ndts;
				// ndts = ndtLocalMap_->pseudoTransformNDT(T);
				double score = 1;
				if(ndts.size() == 0) fprintf(stderr, "ERROR no gaussians in measurement!!!\n");
				for(int n = 0; n < ndts.size(); n++){
					Eigen::Vector3d m = T * ndts[n]->getMean();
					//if(m[2] < zfilt_min) continue;
					perception_oru::NDTCell *cell;
					pcl::PointXYZ p;
					p.x = m[0]; p.y = m[1]; p.z = m[2];
					if(ndtMap->getCellAtPoint(p, cell)){
						if(cell == NULL) continue;
						if(cell->hasGaussian_){
							Eigen::Matrix3d covCombined = cell->getCov() + T.rotation() * ndts[n]->getCov() * T.rotation().transpose();
							Eigen::Matrix3d icov;
							bool exists;
							double det = 0;
							covCombined.computeInverseAndDetWithCheck(icov, det, exists);
							if(!exists) continue;
							double l = (cell->getMean() - m).dot(icov * (cell->getMean() - m));
							if(l * 0 != 0) continue;
							score += 0.1 + 0.9 * exp(-0.5*l / 2.0);
						}else{

						}
					}
				}
			particleCloud[i].SetLikelihood(score);
		}
		for(unsigned int j = 0; j < ndts.size(); j++)
			delete ndts[j];

		Normalize();
		if(forceSIR)
			SIRUpdate();
		else{
			double varP = 0;
			for(int i = 0; i < particleCloud.size(); i++)
				varP += (particleCloud[i].GetProbability() - 1.0 / double(particleCloud.size()))
				        * (particleCloud[i].GetProbability() - 1.0 / double(particleCloud.size()));
			varP /= double(particleCloud.size());
			varP = sqrt(varP);
			if(varP > varLimit || sinceSIR > sirCount){
				//fprintf(stderr,"-SIR- ");
				sinceSIR = 0;
				SIRUpdate();
			}else
				sinceSIR++;
		}
	}
}


void perception_oru::particle_filter::Predict2D(double x, double y, double th, double sx, double sy, double sth){
	float dx = 0.0, dy = 0.0, dl = 0.0;
	float t = 0.0;
	float dxe, dye; ///<estimates
	std::default_random_engine generator;
	float eps=0.000001;
	for(int i = 0; i < particleCloud.size(); i++){
		double px, py, pz, pr, pp, pt;
		particleCloud[i].GetXYZ(px, py, pz);
		particleCloud[i].GetRPY(pr, pp, pt);

		std::normal_distribution<double> distribution_x(0, sx);
		std::normal_distribution<double> distribution_y(0, sy);
		std::normal_distribution<double> distribution_th(0, sth);
		///Generate noise from normal distribution
		dxe = x + distribution_x(generator);
		dye = y + distribution_y(generator);

		//if(fabs(x)<eps && fabs(y)<eps){
		//	dxe = x*0;
		//	dye = y*0;
		//
		//}


		dl = sqrt(dxe * dxe + dye * dye);

		//if(fabs(dxe)<eps && fabs(dye)<eps)
		//t = 0;
		//else
		t = atan2(dye, dxe);

		dx = dl * cos(pt + t);
		dy = dl * sin(pt + t);
		//ROS_INFO_STREAM("th"<<th);
		px += dx;
		py += dy;
		pt = pt + th + distribution_th(generator);
		toPI(pt);
		particleCloud[i].Set(pr, pp, pt, px, py, pz);
	}
	//    isAvgSet = false;
}
void perception_oru::particle_filter::to2PI(double &a){
	a = (double)fmod((double)(a), (double)( 2 * M_PI));
	if(a < 0) a += 2 * (double)M_PI;
}
void perception_oru::particle_filter::toPI(double &a){
	if(a > M_PI)
		while(a > M_PI) a -= 2.0 * M_PI;
	else
	if(a < -M_PI)
		while(a < -M_PI) a += 2.0 * M_PI;
}

void perception_oru::particle_filter::Normalize(){
	int i;
	double summ = 0;
	double sumX = 0, sumY = 0;
	double sumW = 0;
	double ax = 0, ay = 0;

	//isAvgSet = false;
	for(i = 0; i < particleCloud.size(); i++){
		particleCloud[i].SetProbability(particleCloud[i].GetProbability() * particleCloud[i].GetLikelihood());
		summ += particleCloud[i].GetProbability();
	}
	if(summ != 0){
		for(i = 0; i < particleCloud.size(); i++){
			particleCloud[i].SetProbability(particleCloud[i].GetProbability() / summ);
			weights[i]=particleCloud[i].GetProbability();
			double x, y, z, r, p, t;
			particleCloud[i].GetXYZ(x, y, z);
			particleCloud[i].GetRPY(r, p, t);
			sumX += particleCloud[i].GetProbability() * x;
			sumY += particleCloud[i].GetProbability() * y;
			ax += particleCloud[i].GetProbability() * cos(t);
			ay += particleCloud[i].GetProbability() * sin(t);
			sumW += particleCloud[i].GetProbability();
		}
	}
	else{
		for(i = 0; i < particleCloud.size(); i++){
			particleCloud[i].SetProbability(1.0 / particleCloud.size());
			weights[i]=particleCloud[i].GetProbability();
			double x, y, z, r, p, t;
			particleCloud[i].GetXYZ(x, y, z);
			particleCloud[i].GetRPY(r, p, t);
			sumX += particleCloud[i].GetProbability() * x;
			sumY += particleCloud[i].GetProbability() * y;
			ax += particleCloud[i].GetProbability() * cos(t);
			ay += particleCloud[i].GetProbability() * sin(t);
			sumW += particleCloud[i].GetProbability();
		}
	}
		m_pose << sumX, sumY, atan2(ay, ax);
		std::sort (weights.begin(), weights.end(),myfunction);
		//ROS_INFO_STREAM(weights.front()<<" "<< weights.back());
}
void perception_oru::particle_filter::SIRUpdate(){
	std::vector<particle> tmp2;
	std::default_random_engine generator;
	std::uniform_real_distribution<double> dist(0, 1);
	double U = 0, Q = 0;
	int i = 0, j = 0, k = 0;
	U = dist(generator) / (double)particleCloud.size();
	//fprintf(stderr,"SIRUpdate()::U=%.6f\n",U);
	while(U < 1.0){
		if(Q > U){
			U += 1.0 / (double)particleCloud.size();
			if(k >= particleCloud.size() || i >= particleCloud.size()){
				while(i < particleCloud.size()){
					tmp[i] = particleCloud[particleCloud.size() - 1];
					tmp[i].probability = 1.0 / (double)particleCloud.size();
					i++;
				}
				//fprintf(stderr,"ERROR: SIRupdate:: Invalid index k='%d' or i='%d'\n",k,i);
				tmp2 = particleCloud;
				particleCloud = tmp;
				tmp = tmp2;
				return;
			}
			tmp[i] = particleCloud[k];
			tmp[i].probability = 1.0 / (double)particleCloud.size();
			i++;
		}else {
			j++;
			k = j;
			if(j >= particleCloud.size()){
				while(i < particleCloud.size()){
					tmp[i] = particleCloud[particleCloud.size() - 1];
					tmp[i].probability = 1.0 / (double)particleCloud.size();
					i++;
				}
				//fprintf(stderr,"ERROR: SIRupdate:: Invalid index j='%d' \n",j);
				tmp2 = particleCloud;
				particleCloud = tmp;
				tmp = tmp2;
				return;
			}
			Q += particleCloud[j].probability;
			///j++; ///WAS HERE until 30.7.2008

			if(j == particleCloud.size()){
				while(i < particleCloud.size()){
					tmp[i] = particleCloud[k - 1];
					tmp[i].probability = 1.0 / (double)particleCloud.size();
					i++;
				}
				tmp2 = particleCloud;
				particleCloud = tmp;
				tmp = tmp2;
				return;
			}
		}
	} //While
	while(i < particleCloud.size()){
		if(k >= particleCloud.size()) k = particleCloud.size() - 1;
		tmp[i] = particleCloud[k];
		tmp[i].probability = 1.0 / (double)particleCloud.size();
		i++;
	}
	//  isAvgSet = false;
	tmp2 = particleCloud;
	particleCloud = tmp;
	tmp = tmp2;
}

void perception_oru::particle_filter::GetRandomPoint(perception_oru::NDTCell* cell, double &x, double &y, double &th){
	Eigen::Vector3d mean = cell->getMean();
	Eigen::Matrix3d cov = cell->getCov();
	double mX = mean[0], mY = mean[1], mTh = mean[2];
	double varX = cov(0, 0), varY = cov(1, 1), varTh = cov(2, 2);
	std::default_random_engine generator;

	std::normal_distribution<double> distribution_x(mX, varX);
	std::normal_distribution<double> distribution_y(mY, varY);
	std::normal_distribution<double> distribution_th(mTh, varTh);
	x = distribution_x(generator);
	y = distribution_y(generator);
	th = distribution_th(generator);
}

void perception_oru::particle_filter::EigenSort( Eigen::Vector3d &eigenvalues, Eigen::Matrix3d &eigenvectors ){
	int k, j, i;
	double p;

	for(i = 0; i < 2; i++){
		p = eigenvalues(k = i);
		for(j = i + 1; j < 3; j++)
			if(fabs(eigenvalues(j)) >= fabs(p))
				p = eigenvalues(k = j);
		if(k != i){
			eigenvalues.row(k).swap(eigenvalues.row(i));
			eigenvectors.col(k).swap(eigenvectors.col(i));
		}
	}
}

Eigen::Affine3d perception_oru::particle_filter::getAsAffine(float x, float y, float yaw ){
	Eigen::Matrix3d m;

	m = Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX())
	    * Eigen::AngleAxisd(0, Eigen::Vector3d::UnitY())
	    * Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ());
	Eigen::Translation3d v(x, y, 0);
	Eigen::Affine3d T = Eigen::Affine3d::Identity();
	T.rotate(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
	T.translation()<<x,y,0;
	return T;
}
