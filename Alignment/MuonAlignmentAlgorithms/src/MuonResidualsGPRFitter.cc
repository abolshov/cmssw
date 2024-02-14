#ifdef STANDALONE_FITTER
#include "MuonResidualsGPRFitter.h"
#else
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsGPRFitter.h"
#endif
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "TMath.h"
#include "TH1.h"
// #include "TF1.h"
// #include "TVector2.h"
#include "TFile.h"
#include "TTree.h"
#include "TH2F.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TMarker.h"
#include "TCanvas.h"
#include "TGraph2D.h"
// #include "TRobustEstimator.h"
// #include "Math/MinimizerOptions.h"

#include <map>
#include <iomanip>
// #include <functional>
#include <algorithm>
#include <iterator>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <memory>
#include <fstream>
#include <stdexcept>

#include "Alignment/MuonAlignmentAlgorithms/interface/Tracer.hpp"
#include "Alignment/CommonAlignment/interface/Utilities.h"

#include "TMatrixDSym.h"

static TMinuit* MuonResidualsGPRFitter_TMinuit;

static double fcn_zero = 0.0;
static double avg_call_duration = 0.0;
static int call_id = 0;

using DT_6DOF = MuonResidualsGPRFitter::DataDT_6DOF;
using DT_5DOF = MuonResidualsGPRFitter::DataDT_5DOF;
using CSC_6DOF = MuonResidualsGPRFitter::DataCSC_6DOF;
using PARAMS = MuonResidualsGPRFitter::PARAMS;

namespace 
{
    TMinuit* minuit;
}

double MuonResidualsGPRFitter_logPureGaussian(double residual, double center, double sigma) 
{
    sigma = fabs(sigma);
    static const double cgaus = 0.5 * log(2. * M_PI);
    return (-pow(residual - center, 2) * 0.5 / sigma / sigma) - cgaus - log(sigma);
}

MuonResidualsGPRFitter::MuonResidualsGPRFitter()
    : m_DTGeometry(nullptr),
      m_CSCGeometry(nullptr),
      m_printLevel(0),
      m_strategy(0),
      m_value(npar(), 0.0),
      m_error(npar(), 0.0),
      m_loglikelihood(0.0) {}

MuonResidualsGPRFitter::MuonResidualsGPRFitter(DTGeometry const* DTGeom, CSCGeometry const* CSCGeom, std::vector<std::string> const& opt)
    : m_DTGeometry(DTGeom),
      m_CSCGeometry(CSCGeom),
      m_printLevel(0),
      m_strategy(2),
      m_value(npar(), 0.0),
      m_error(npar(), 0.0),
      m_loglikelihood(0.0),
      m_options(opt) {}

// if something happens comment out m_data, destructor and make it default
MuonResidualsGPRFitter::~MuonResidualsGPRFitter()
{
    if (!m_data.empty())
    {
        for (auto mapIt: m_data)
        {
            for (auto ptr: mapIt.second)
            {
                delete [] ptr;
            }
        }
    }
}

// NEW FCN START
struct ResidData
{
    double dx;
    double dy;
    double dz;
    double dphix;
    double dphiy;
    double dphiz;
    double track_x;
    double track_y;
    double track_dxdz;
    double track_dydz;
    double slope = 0.0;
    double alpha = 0.0;
};

// functions to compute DT residuals
inline double ResidX(ResidData&& res_data)
{
    auto&& [dx, dy, dz, dphix, dphiy, dphiz, track_x, track_y, track_dxdz, track_dydz, slope, alpha] = res_data;
    return dx - track_dxdz*dz - track_y*track_dxdz*dphix + track_x*track_dxdz*dphiy - track_y*dphiz + slope*alpha;
}

inline double ResidY(ResidData&& res_data)
{
    auto&& [dx, dy, dz, dphix, dphiy, dphiz, track_x, track_y, track_dxdz, track_dydz, slope, alpha] = res_data;
    return dy - track_dydz*dz - track_y*track_dydz*dphix + track_x*track_dydz*dphiy + track_x*dphiz + slope*alpha;
}

inline double ResidDXDZ(ResidData&& res_data)
{
    auto&& [dx, dy, dz, dphix, dphiy, dphiz, track_x, track_y, track_dxdz, track_dydz, slope, alpha] = res_data;
    return -track_dxdz*track_dydz*dphix + (1.0 + track_dxdz*track_dxdz)*dphiy - track_dydz*dphiz;
}

inline double ResidDYDZ(ResidData&& res_data)
{
    auto&& [dx, dy, dz, dphix, dphiy, dphiz, track_x, track_y, track_dxdz, track_dydz, slope, alpha] = res_data;
    return -(1.0 + track_dydz * track_dydz)*dphix + track_dxdz*track_dydz*dphiy + track_dxdz*dphiz;
}

// functions to compute CSC residuals
inline double Resid(ResidData&& res_data, double R)
{
    auto&& [dx, dy, dz, dphix, dphiy, dphiz, track_x, track_y, track_dxdz, track_dydz, slope, alpha] = res_data;
    return dx - (track_x/R - 3.0*(track_x/R)*(track_x/R)*(track_x/R))*dy - track_dxdz*dz - track_y*track_dxdz*dphix + track_x*track_dxdz*dphiy - track_y*dphiz +
            slope*alpha;
}

inline double ResidSlope(ResidData&& res_data, double R)
{
    auto&& [dx, dy, dz, dphix, dphiy, dphiz, track_x, track_y, track_dxdz, track_dydz, slope, alpha] = res_data;
    return -0.5*track_dxdz/R*dy + (track_x/R - track_dxdz*track_dydz)*dphix + (1.0 + track_dxdz*track_dxdz)*dphiy - track_dydz*dphiz;
}

// input format: 
// angle[0] = dphiz; 
// angle[1] = dphiy;
// angle[2] = dphix;
// Reason: rotaions are executed from RIGHT to LEFT (bc of matrix multiplication)
// first rotate around X by angle[2]
// than rotate around Y by anlge[1]
// lastly rotate around Z by angle[0]
// https://en.wikipedia.org/wiki/Euler_angles#Conversion_to_other_orientation_representations
// https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
align::RotationType MatrixXYZ(align::EulerAngles const& angles)
{
    // Scalar is just a typedef for double
    align::Scalar s1 = std::sin(angles[0]), c1 = std::cos(angles[0]);
    align::Scalar s2 = std::sin(angles[1]), c2 = std::cos(angles[1]);
    align::Scalar s3 = std::sin(angles[2]), c3 = std::cos(angles[2]);

    // in constructor(fisrt row of the matrix,     xx xy xz
    //                second row of the matrix,    yx yy yz
    //                third row of the matrix)     zx zy zz
    return align::RotationType(c1*c2, c1*s2*s3 - c3*s1, s1*s3 + c1*c3*s2,
                               c2*s1, c1*c3 + s1*s2*s3, c3*s1*s2 - c1*s3,
                               -s2, c2*s3, c2*c3);
}

align::RotationType MatrixXYZ(double dphix, double dphiy, double dphiz)
{
    align::EulerAngles angles(3);
    angles[0] = dphiz;
    angles[1] = dphiy;
    angles[2] = dphix;

    return MatrixXYZ(angles);
}

// Is designed to handled SMALL ANGLES ONLY!
// small angle: sin < 0.1
// Also in case when non-unique solution (gimbal lock) some angles will be forced o be pi/2, so this case should throw
// https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
align::EulerAngles AnglesXYZ(align::RotationType const& matrix)
{
    if (std::abs(matrix.zx()) >= 1) throw std::logic_error("Gimbal lock detected: ambiguous solition");
    
    align::EulerAngles res(3);
    align::Scalar dphiy = std::asin(-1.0*matrix.zx());
    if (std::abs(dphiy) > 0.1) throw std::out_of_range("Small angle approximation violated");
    align::Scalar dphiz = std::atan2(matrix.yx(), matrix.xx());
    align::Scalar dphix = std::atan2(matrix.zy(), matrix.zz());

    res[0] = dphix;
    res[1] = dphiy;
    res[2] = dphiz;

    return res;
}

void GlobalFitter_FCN(int &npar, double *gin, double &fval, double *par, int iflag) 
{
    std::cout << std::setprecision(8);
    ++call_id;
    MuonResidualsGPRFitterFitInfo *fitinfo = (MuonResidualsGPRFitterFitInfo *)(minuit->GetObjectFit());
    MuonResidualsGPRFitter *gpr_fitter = fitinfo->gpr_fitter();
    DTGeometry const* DTgeom = gpr_fitter->GetDTGeometry();
    CSCGeometry const* CSCgeom = gpr_fitter->GetCSCGeometry();

    fval = 0.0;

    if (gpr_fitter->IsEmpty()) return;

    using namespace std::chrono;
    auto start = high_resolution_clock::now();

    for (auto chamber_data = gpr_fitter->DataBegin(); chamber_data != gpr_fitter->DataEnd(); ++chamber_data)
    {
        auto const& [id, resid_vec] = *chamber_data;

        if (id.subdetId() == MuonSubdetId::DT && DTgeom) 
        {
            Surface::PositionType position = DTgeom->idToDet(id)->position();
            Surface::RotationType orientation = DTgeom->idToDet(id)->rotation();

            DTChamberId dtId(id.rawId());
            int station = dtId.station();
                         
            // transform rotation angle from global to local
            // input format: 
            // globAngles[0] = dphiz; 
            // globAngles[1] = dphiy;
            // globAngles[2] = dphix;
            // Reason: rotaions are executed from RIGHT to LEFT (bc of matrix multiplication)
            // first rotate around X by globAngles[2]
            // than rotate around Y by globAngles[1]
            // lastly rotate around Z by globAngles[0]
            
            double dphix = par[static_cast<int>(PARAMS::kAlignPhiX)];
            double dphiy = par[static_cast<int>(PARAMS::kAlignPhiY)];
            double dphiz = par[static_cast<int>(PARAMS::kAlignPhiZ)];

            // align::RotationType glob_rot = MatrixXYZ(globAngles); // proper global gpr matrix in Z(3)Y(2)X(1) convection; 
            align::RotationType glob_rot = MatrixXYZ(dphix, dphiy, dphiz);
            align::RotationType loc_rot = orientation*glob_rot*orientation.transposed();
            align::EulerAngles locAngles = AnglesXYZ(loc_rot);

            // transformed angles
            double alignphix_l = locAngles[0];
            double alignphiy_l = locAngles[1];
            double alignphiz_l = locAngles[2]; 

            // transform translations from global to local
            double dx, dy, dz;
            dx = glob_rot.xx()*position.x() + glob_rot.xy()*position.y() + glob_rot.xz()*position.z() - position.x();
            dy = glob_rot.yx()*position.x() + glob_rot.yy()*position.y() + glob_rot.yz()*position.z() - position.y();
            dz = glob_rot.zx()*position.x() + glob_rot.zy()*position.y() + glob_rot.zz()*position.z() - position.z();

            double alignx_g = par[static_cast<int>(PARAMS::kAlignX)] + dx;
            double aligny_g = par[static_cast<int>(PARAMS::kAlignY)] + dy;
            double alignz_g = par[static_cast<int>(PARAMS::kAlignZ)] + dz;

            GlobalVector translation_g = GlobalVector(alignx_g, aligny_g, alignz_g);
            LocalVector translation_l = DTgeom->idToDet(id)->toLocal(translation_g);

            // transformed shifts
            double alignx_l = translation_l.x(); 
            double aligny_l = translation_l.y(); 
            double alignz_l = translation_l.z();

            std::vector<double> sigmas = gpr_fitter->GetStdDevs(id);
            double resXsigma = sigmas[static_cast<int>(DT_6DOF::kResidX)];
            double resYsigma = sigmas[static_cast<int>(DT_6DOF::kResidY)];
            double slopeXsigma = sigmas[static_cast<int>(DT_6DOF::kResSlopeX)];
            double slopeYsigma = sigmas[static_cast<int>(DT_6DOF::kResSlopeY)];

            // std::vector<double> means = gpr_fitter->GetMeans(id);
            // double resXmean = means[static_cast<int>(DT_6DOF::kResidX)];
            // double resYmean = means[static_cast<int>(DT_6DOF::kResidY)];
            // double slopeXmean = means[static_cast<int>(DT_6DOF::kResSlopeX)];
            // double slopeYmean = means[static_cast<int>(DT_6DOF::kResSlopeY)];

            if (station != 4)
            {
                std::vector<double*>::const_iterator resid_begin = resid_vec.cbegin();
                std::vector<double*>::const_iterator resid_end = resid_vec.cend();
                std::vector<double*>::const_iterator it = resid_begin;
                for (it = resid_begin; it != resid_end; ++it)
                {
                    double residX = (*it)[static_cast<int>(DT_6DOF::kResidX)];
                    double residY = (*it)[static_cast<int>(DT_6DOF::kResidY)];
                    double resslopeX = (*it)[static_cast<int>(DT_6DOF::kResSlopeX)];
                    double resslopeY = (*it)[static_cast<int>(DT_6DOF::kResSlopeY)];
                    double positionX = (*it)[static_cast<int>(DT_6DOF::kPositionX)];
                    double positionY = (*it)[static_cast<int>(DT_6DOF::kPositionY)];
                    double angleX = (*it)[static_cast<int>(DT_6DOF::kAngleX)];
                    double angleY = (*it)[static_cast<int>(DT_6DOF::kAngleY)];

                    double residXpeak = ResidX({alignx_l, aligny_l, alignz_l, alignphix_l, alignphiy_l, alignphiz_l, positionX, positionY, angleX, angleY, resslopeX});
                    double residYpeak = ResidY({alignx_l, aligny_l, alignz_l, alignphix_l, alignphiy_l, alignphiz_l, positionX, positionY, angleX, angleY, resslopeY});
                    double slopeXpeak = ResidDXDZ({alignx_l, aligny_l, alignz_l, alignphix_l, alignphiy_l, alignphiz_l, positionX, positionY, angleX, angleY});
                    double slopeYpeak = ResidDYDZ({alignx_l, aligny_l, alignz_l, alignphix_l, alignphiy_l, alignphiz_l, positionX, positionY, angleX, angleY});

                    double weight = 1.0;

                    fval += -weight * MuonResidualsGPRFitter_logPureGaussian(residX, residXpeak, resXsigma);
                    fval += -weight * MuonResidualsGPRFitter_logPureGaussian(residY, residYpeak, resYsigma);
                    fval += -weight * MuonResidualsGPRFitter_logPureGaussian(resslopeX, slopeXpeak, slopeXsigma);
                    fval += -weight * MuonResidualsGPRFitter_logPureGaussian(resslopeY, slopeYpeak, slopeYsigma);
                }
            }
            else 
            {
                continue;
            }
        }

        if (id.subdetId() == MuonSubdetId::CSC && CSCgeom)
        {
            Surface::PositionType position = CSCgeom->idToDet(id)->position();
            Surface::RotationType orientation = CSCgeom->idToDet(id)->rotation();

            double csc_x = position.x();
            double csc_y = position.y();

            double csc_R = sqrt(csc_x*csc_x + csc_y*csc_y);

            // transform rotation angle from global to local
            double dphix = par[static_cast<int>(PARAMS::kAlignPhiX)];
            double dphiy = par[static_cast<int>(PARAMS::kAlignPhiY)];
            double dphiz = par[static_cast<int>(PARAMS::kAlignPhiZ)];

            // align::RotationType glob_rot = MatrixXYZ(globAngles); // proper global gpr matrix in Z(3)Y(2)X(1) convection; 
            align::RotationType glob_rot = MatrixXYZ(dphix, dphiy, dphiz);
            align::RotationType loc_rot = orientation*glob_rot*orientation.transposed();
            align::EulerAngles locAngles = AnglesXYZ(loc_rot);

            // transformed angles
            double alignphix_l = locAngles[0];
            double alignphiy_l = locAngles[1];
            double alignphiz_l = locAngles[2]; 

            // transform translations from global to local
            double dx, dy, dz;
            dx = glob_rot.xx()*position.x() + glob_rot.xy()*position.y() + glob_rot.xz()*position.z() - position.x();
            dy = glob_rot.yx()*position.x() + glob_rot.yy()*position.y() + glob_rot.yz()*position.z() - position.y();
            dz = glob_rot.zx()*position.x() + glob_rot.zy()*position.y() + glob_rot.zz()*position.z() - position.z();

            double alignx_g = par[static_cast<int>(PARAMS::kAlignX)] + dx;
            double aligny_g = par[static_cast<int>(PARAMS::kAlignY)] + dy;
            double alignz_g = par[static_cast<int>(PARAMS::kAlignZ)] + dz;

            GlobalVector translation_g = GlobalVector(alignx_g, aligny_g, alignz_g);
            LocalVector translation_l = CSCgeom->idToDet(id)->toLocal(translation_g);

            // transformed shifts
            double alignx_l = translation_l.x(); 
            double aligny_l = translation_l.y(); 
            double alignz_l = translation_l.z();

            // double resSigma = 0.5;
            // double resSlopeSigma = 0.002;
            std::vector<double> sigmas = gpr_fitter->GetStdDevs(id);
            double resSigma = sigmas[static_cast<int>(CSC_6DOF::kResid)];
            double resSlopeSigma = sigmas[static_cast<int>(CSC_6DOF::kResSlope)];
            
            std::vector<double*>::const_iterator resid_begin = resid_vec.cbegin();
            std::vector<double*>::const_iterator resid_end = resid_vec.cend();
            std::vector<double*>::const_iterator it = resid_begin;
            for (it = resid_begin; it != resid_end; ++it)
            {
                double resid = (*it)[static_cast<int>(CSC_6DOF::kResid)];
                double resSlope = (*it)[static_cast<int>(CSC_6DOF::kResSlope)];
                double positionX = (*it)[static_cast<int>(CSC_6DOF::kPositionX)];
                double positionY = (*it)[static_cast<int>(CSC_6DOF::kPositionY)];
                double angleX = (*it)[static_cast<int>(CSC_6DOF::kAngleX)];
                double angleY = (*it)[static_cast<int>(CSC_6DOF::kAngleY)];

                double residPeak = Resid({alignx_l, aligny_l, alignz_l, alignphix_l, alignphiy_l, alignphiz_l, positionX, positionY, angleX, angleY}, csc_R);
                double resSlopePeak = ResidSlope({alignx_l, aligny_l, alignz_l, alignphix_l, alignphiy_l, alignphiz_l, positionX, positionY, angleX, angleY}, csc_R);

                double weight = 1.0;

                fval += -weight * MuonResidualsGPRFitter_logPureGaussian(resid, residPeak, resSigma);
                fval += -weight * MuonResidualsGPRFitter_logPureGaussian(resSlope, resSlopePeak, resSlopeSigma);
            }
        }  
    }
    
    if (call_id > 1) fval -= fcn_zero;

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    std::cout << "******************************************" << "\n";
    std::cout << "FCN call #: " << call_id << "\n";
    std::cout << "execution time " << duration.count() << " ms" << "\n";
    std::cout << "FCN = " << fval << "\n";
    std::cout << "global params: ";
    for (int i = 0; i < npar; ++i)
    {
        std::cout << par[i] << " ";
    } 
    std::cout << "\n";
    std::cout << "******************************************" << "\n";
    avg_call_duration += static_cast<double>(duration.count());
}
// NEW FCN END

void MuonResidualsGPRFitter::inform(TMinuit* tMinuit) { minuit = tMinuit; }

bool MuonResidualsGPRFitter::dofit(void (*fcn)(int &, double *, double &, double *, int),
                                   std::vector<int> &parNum,
                                   std::vector<std::string> &parName,
                                   std::vector<double> &start,
                                   std::vector<double> &step,
                                   std::vector<double> &low,
                                   std::vector<double> &high) 
{
    MuonResidualsGPRFitterFitInfo *fitinfo = new MuonResidualsGPRFitterFitInfo(this);

    // configure Minuit object
    MuonResidualsGPRFitter_TMinuit = new TMinuit(npar());
    MuonResidualsGPRFitter_TMinuit->SetPrintLevel();
    MuonResidualsGPRFitter_TMinuit->SetObjectFit(fitinfo);
    MuonResidualsGPRFitter_TMinuit->SetFCN(fcn);
    inform(MuonResidualsGPRFitter_TMinuit);

    std::vector<int>::const_iterator iNum = parNum.begin();
    std::vector<std::string>::const_iterator iName = parName.begin();
    std::vector<double>::const_iterator istart = start.begin();
    std::vector<double>::const_iterator istep = step.begin();
    std::vector<double>::const_iterator ilow = low.begin();
    std::vector<double>::const_iterator ihigh = high.begin();

    for (; iNum != parNum.end(); ++iNum, ++iName, ++istart, ++istep, ++ilow, ++ihigh) 
    {
        MuonResidualsGPRFitter_TMinuit->DefineParameter(*iNum, iName->c_str(), *istart, *istep, *ilow, *ihigh);
        if (*iNum > 5)
        {
            MuonResidualsGPRFitter_TMinuit->FixParameter(*iNum);
        }
    }

    Tracer::instance() << "start: ";
    Tracer::instance().copy(start.begin(), start.end());
    Tracer::instance() << "\n";

    Double_t fmin, fedm, errdef;
    Int_t npari, nparx, istat;

    double arglist[10];
    int ierflg;
    int smierflg;  //second MIGRAD ierflg

    // chi^2 errors should be 1.0, log-likelihood should be 0.5
    for (int i = 0; i < 10; i++)
    {
        arglist[i] = 0.0;
    }
    // arglist[0] = 70.64402; // 99% CL
    arglist[0] = 13.068225; // 70% CL
    ierflg = 0;
    smierflg = 0;
    MuonResidualsGPRFitter_TMinuit->mnexcm("SET ERR", arglist, 1, ierflg);
    if (ierflg != 0) 
    {
        delete MuonResidualsGPRFitter_TMinuit;
        delete fitinfo;
        return false;
    }

    // set strategy = 2 (more refined fits)
    for (int i = 0; i < 10; i++)
    {
        arglist[i] = 0.0;
    }
    arglist[0] = m_strategy;
    ierflg = 0;
    MuonResidualsGPRFitter_TMinuit->mnexcm("SET STR", arglist, 1, ierflg);
    if (ierflg != 0) 
    {
        delete MuonResidualsGPRFitter_TMinuit;
        delete fitinfo;
        return false;
    }

    int ndim = npar();
    double par[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int flag = 1;
    double grad[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    MuonResidualsGPRFitter_TMinuit->Eval(ndim, grad, fcn_zero, par, flag);

    double true_shift[6] = {0.0, 0.0, 0.0, -0.001, 0.002, -0.003};

    double min_fcn = 0.0;
    MuonResidualsGPRFitter_TMinuit->Eval(ndim, grad, min_fcn, true_shift, flag);
    // std::cout << std::setprecision(10) << "FCN(true_shift) = " << min_fcn << "\n";
    Tracer::instance() << std::setprecision(10) << "FCN(true_shift) = " << min_fcn << "\n";
    // min_fcn = 0.0;
    // MuonResidualsGPRFitter_TMinuit->Eval(ndim, grad, min_fcn, calc_shift, flag);
    // std::cout << std::setprecision(10) << "FCN(calc_shift) = " << min_fcn << "\n";

    bool try_again = false;

    // minimize
    for (int i = 0; i < 10; i++)
    {
        arglist[i] = 0.0;
    }
    arglist[0] = 50000;
    ierflg = 0;
    MuonResidualsGPRFitter_TMinuit->mnexcm("MIGRAD", arglist, 1, ierflg);
    MuonResidualsGPRFitter_TMinuit->mnstat(fmin, fedm, errdef, npari, nparx, istat);

    Tracer::instance() << "MIGRAD:\n"
                       << "\tfmin = " << fmin << "\n"
                       << "\tfedm = " << fedm << "\n"
                       << "\tistat = " << istat << "\n"
                       << "\tierflg = " << ierflg << "\n";
                       
    if (ierflg != 0)
    {
        try_again = true;
    }
    // just once more, if needed (using the final Minuit parameters from the failed fit; often works)
    if (try_again) 
    {
        for (int i = 0; i < 10; i++)
        {
            arglist[i] = 0.0;
        }
        arglist[0] = 50000;
        arglist[1] = 0.05;
        smierflg = 0;
        MuonResidualsGPRFitter_TMinuit->mnexcm("MIGRAD", arglist, 2, smierflg);
    }

    for (int i = 0; i < 10; i++)
    {
        arglist[i] = 0.0;
    }
    ierflg = 0;
    MuonResidualsGPRFitter_TMinuit->mnexcm("HESSE", arglist, 0, ierflg);
    MuonResidualsGPRFitter_TMinuit->mnstat(fmin, fedm, errdef, npari, nparx, istat);

    Tracer::instance() << "HESSE:\n"
                       << "\tfmin = " << fmin << "\n"
                       << "\tfedm = " << fedm << "\n"
                       << "\tistat = " << istat << "\n"
                       << "\tierflg = " << ierflg << "\n";

    // read-out the results
    m_loglikelihood = -fmin;

    m_value.clear();
    m_error.clear();
    for (int i = 0; i < npar(); i++) 
    {
        double v, e;
        MuonResidualsGPRFitter_TMinuit->GetParameter(i, v, e);
        m_value.push_back(v);
        m_error.push_back(e);
    }

    min_fcn = 0.0;
    MuonResidualsGPRFitter_TMinuit->Eval(ndim, grad, min_fcn, m_value.data(), flag);
    Tracer::instance() << std::setprecision(10) << "FCN(calc_shift) = " << min_fcn << "\n";

    Tracer::instance() << "m_value: ";
    Tracer::instance().copy(m_value.begin(), m_value.end());
    Tracer::instance() << "\n";
    Tracer::instance() << "m_error: ";
    Tracer::instance().copy(m_error.begin(), m_error.end());
    Tracer::instance() << "\n";

    avg_call_duration /= call_id;
    Tracer::instance() << "average call duration = " << avg_call_duration << " ms\n";
    Tracer::instance() << "# FCN calls = " << call_id << "\n";

    delete MuonResidualsGPRFitter_TMinuit;
    delete fitinfo;
    if (smierflg != 0)
    {
        return false;
    }    
    return true;
}

bool MuonResidualsGPRFitter::Fit()
{
    std::vector<int> nums{ static_cast<int>(PARAMS::kAlignX),
                           static_cast<int>(PARAMS::kAlignY),
                           static_cast<int>(PARAMS::kAlignZ),
                           static_cast<int>(PARAMS::kAlignPhiX),
                           static_cast<int>(PARAMS::kAlignPhiY),
                           static_cast<int>(PARAMS::kAlignPhiZ) };

    std::vector<std::string> names{ "AlignX",
                                    "AlignY",
                                    "AlignZ",
                                    "AlignPhiX",
                                    "AlignPhiY",
                                    "AlignPhiZ" };

    // std::random_device rd;
    // std::uniform_real_distribution<double> dist(-0.001, 0.001);
    // auto randNum = [&rd, &dist]() { return dist(rd); };
    // std::vector<double> starts(6);
    // std::generate(starts.begin(), starts.end(), randNum);

    std::vector<double> starts{ 0.01,
                                0.01,
                                0.01,
                                0.0001,
                                0.0001,
                                0.0001 };

    std::vector<double> steps{ 0.002,
                               0.002,
                               0.002,
                               0.00002,
                               0.00002,
                               0.00002 };

    // parameter ranges 
    std::vector<double> lows{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    std::vector<double> highs{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    return dofit(&GlobalFitter_FCN, nums, names, starts, steps, lows, highs);
}

void MuonResidualsGPRFitter::PlotFCN(int grid_size, std::vector<double> const& lows, std::vector<double> const& highs)
{
    MuonResidualsGPRFitterFitInfo* fitinfo = new MuonResidualsGPRFitterFitInfo(this);

    // configure Minuit object
    TMinuit* MuonResidualsGPRFitter_TMinuit = new TMinuit(npar());
    MuonResidualsGPRFitter_TMinuit->SetPrintLevel(-1);
    MuonResidualsGPRFitter_TMinuit->SetObjectFit(fitinfo);
    MuonResidualsGPRFitter_TMinuit->SetFCN(GlobalFitter_FCN);
    inform(MuonResidualsGPRFitter_TMinuit);

    std::vector<std::string> par_names = {"dx", "dy", "dz", "dphix", "dphiy", "dphiz"};
    std::vector<double> steps;

    int ndim = lows.size();
    if (grid_size % 2 != 0) ++grid_size;

    for (int i = 0; i < ndim; ++i)
    {
        steps.push_back((highs[i] - lows[i])/grid_size);
    }

    for (int i = 0; i < ndim; ++i)
    {
        MuonResidualsGPRFitter_TMinuit->DefineParameter(i, par_names[i].c_str(), lows[i], steps[i], lows[i], highs[i]);
    }

    std::unique_ptr<TFile> file = std::make_unique<TFile>("fcn_plots.root", "RECREATE");
    std::unique_ptr<TStyle> gStyle = std::make_unique<TStyle>();

    int flag = 1; // only calculate value of likelihood
    std::vector<double> vec_grad(npar(), 0.0);
    double* grad = vec_grad.data();

    if (fcn_zero == 0.0)
    {
        double par[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        int flag = 1;
        MuonResidualsGPRFitter_TMinuit->Eval(ndim, grad, fcn_zero, par, flag);
    }

    // 2d plotting
    std::string name = "graph2d_";
    for (int par_to_plot1 = 0; par_to_plot1 < ndim; ++par_to_plot1)
    {
        for (int par_to_plot2 = par_to_plot1 + 1; par_to_plot2 < ndim; ++par_to_plot2)
        {
            name += par_names[par_to_plot1] + "_vs_" + par_names[par_to_plot2];
            std::vector<double> xvals, yvals, fvals;
            for (int i = 0; i < grid_size + 1; ++i) 
            {
                for (int j = 0; j < grid_size + 1; ++j)
                {
                    double var1 = lows[par_to_plot1] + i*steps[par_to_plot1];
                    double var2 = lows[par_to_plot2] + j*steps[par_to_plot2];
                    std::vector<double> params(ndim);
                    params[par_to_plot1] = var1;
                    params[par_to_plot2] = var2;
                    for (int free_idx = 0; free_idx < ndim; ++free_idx)
                    {
                        if (free_idx != par_to_plot1 && free_idx != par_to_plot2)
                        {
                            params[free_idx] = (highs[free_idx] + lows[free_idx])/2;
                        }
                    }
                    double fval = 0.0;
                    double* par = params.data();
                    MuonResidualsGPRFitter_TMinuit->Eval(ndim, grad, fval, par, flag);
                    xvals.push_back(var1);
                    yvals.push_back(var2);
                    fvals.push_back(fval);
                }
            }

            std::unique_ptr<TGraph2D> graph2d = std::make_unique<TGraph2D>(xvals.size(), xvals.data(), yvals.data(), fvals.data());
            gStyle->SetPalette(1);
            graph2d->Draw("colz");
            file->WriteObject(graph2d.get(), name.c_str());

            name = "graph2d_";
        }
    }

    // 1d plotting
    name = "graph1d_";
    for (int par_to_plot = 0; par_to_plot < ndim; ++par_to_plot)
    {   
        name += par_names[par_to_plot];
        std::vector<double> x, y;
        for (int i = 0; i < grid_size + 1; ++i) 
        {
            double var = lows[par_to_plot] + i*steps[par_to_plot];
            std::vector<double> params(ndim);
            params[par_to_plot] = var;
            for (int free_idx = 0; free_idx < ndim; ++free_idx)
            {
                if (free_idx != par_to_plot)
                {
                    params[free_idx] = (highs[free_idx] + lows[free_idx])/2;
                }
            }
            double fval = 0.0;
            double* par = params.data();
            MuonResidualsGPRFitter_TMinuit->Eval(ndim, grad, fval, par, flag);
            x.push_back(var);
            y.push_back(fval);
        }

        std::unique_ptr<TGraph> graph1d = std::make_unique<TGraph>(x.size(), x.data(), y.data());
        graph1d->SetLineWidth(2);
        graph1d->Draw("AL");
        file->WriteObject(graph1d.get(), name.c_str());

        name = "graph1d_";
    }

    file->Close();
}

size_t MuonResidualsGPRFitter::NTypesOfResid(Alignable const* ali) const
{
    DetId id = ali->geomDetId();
    if (id.subdetId() == MuonSubdetId::CSC) return static_cast<size_t>(DataCSC_6DOF::kNData);
    if (id.subdetId() == MuonSubdetId::DT)
    {
        DTChamberId dtId(id.rawId());
        if (dtId.station() != 4) return static_cast<size_t>(DataDT_6DOF::kNData);
        return static_cast<size_t>(DataDT_5DOF::kNData);
    }
    return 0;
}

size_t MuonResidualsGPRFitter::NTypesOfResid(DetId const& id) const
{
    if (id.subdetId() == MuonSubdetId::CSC) return static_cast<size_t>(DataCSC_6DOF::kNData);
    if (id.subdetId() == MuonSubdetId::DT)
    {
        DTChamberId dtId(id.rawId());
        if (dtId.station() != 4) return static_cast<size_t>(DataDT_6DOF::kNData);
        return static_cast<size_t>(DataDT_5DOF::kNData);
    }
    return 0;
}

void MuonResidualsGPRFitter::CopyData(std::map<Alignable*, MuonResidualsTwoBin*> const& from, double fraction)
{
    for (auto fitIt = from.cbegin(); fitIt != from.cend(); ++fitIt)
    {
        // std::vector<double*> contains many arrays, each of size NTypesOfResid(ali); each array holds residuals calculated from one track
        // in other words, size of each double* is number of different residuals in a chamber (depends on chamber type)
        // size of std::vector<double*> - number of tracks in the chamber from which residuals were calculated

        Alignable* ali = fitIt->first;
        DetId id = ali->geomDetId();

        // don't copy what's not needed
        if (!Select(id)) continue;

        // if (id.subdetId() == MuonSubdetId::DT)
        // {
        //     DTChamberId dtId(id.rawId());
        //     int wheel = dtId.wheel();
        //     int station = dtId.station();
        //     int sector = dtId.sector();

        //     // std::vector<int> allowed{1,4,7,10};
        //     // if (std::find(allowed.begin(), allowed.end(), sector) == allowed.end()) continue;
        //     // if (wheel != 1) continue;
	    //     // if (station != 1) continue;
        //     // if (sector != 7) continue;
        //     std::cout << "Copying from " << wheel << "/" << station << "/" << sector << "\n"; 
        // }

        // sizes of arrays in vector
        size_t resPosSz = fitIt->second->numResidualsPos();
        size_t resNegSz = fitIt->second->numResidualsNeg();

        // select only fraction of residuals
        size_t totResSz = static_cast<size_t>(fraction*resPosSz) + static_cast<size_t>(fraction*resNegSz);
        // std::cout << "\ttotResSz = " << totResSz << "\n";

        if (totResSz == 0) continue;

        // number type
        size_t numResTypes = NTypesOfResid(ali);
        
        // what if at some point for some chamber I cannot allocate memory?
        // option 1: exit from function and use only already copied data
        // option 2: clean already copied data; reduce fraction by e.g. factor of 2; try again; if fails - ...?
        std::vector<double*> tmp(totResSz, nullptr);
        auto size = numResTypes*sizeof(double);
        auto resPosBegin = fitIt->second->residualsPos_begin();
        auto resNegBegin = fitIt->second->residualsNeg_begin();
        for (size_t idx = 0; idx < totResSz; ++idx)
        {
            // allocate memory for this residual
            // allocates 5 or 7 doubles depending on DT/CSC or DT station
            tmp[idx] = new double[numResTypes];
            
            if (idx < resPosSz) 
            {
                std::memcpy(tmp[idx], *(resPosBegin + idx), size);
            }
            else
            {
                size_t negResIdx = idx - resPosSz;
                std::memcpy(tmp[idx], *(resNegBegin + negResIdx), size);
            }
        } 
        m_data.insert({id, tmp});
    }
}

int MuonResidualsGPRFitter::TrackCount() const
{
    int cnt = 0;
    for (auto const& [id, resid_vec]: m_data)
    {
        cnt += resid_vec.size();
    }
    return cnt;
}

void MuonResidualsGPRFitter::ReleaseData()
{
    for (auto& [id, resid_vec]: m_data)
    {
        for (auto res_ptr: resid_vec)
        {
            delete [] res_ptr;
        }
    }

    m_data.clear();
}

void MuonResidualsGPRFitter::Print(int nValues, DetId const& id) const
{
    std::vector<double*> residVec = m_data.at(id);

    if (id.subdetId() == MuonSubdetId::CSC)
    {
        CSCDetId cscId(id.rawId());
        std::cout << cscId << "\n";
        for (int i = 0; i < nValues; ++i)
        {
            std::cout << residVec[i][static_cast<int>(DataCSC_6DOF::kResid)] << " "
                      << residVec[i][static_cast<int>(DataCSC_6DOF::kResSlope)] << " "
                      << residVec[i][static_cast<int>(DataCSC_6DOF::kPositionX)] << " "
                      << residVec[i][static_cast<int>(DataCSC_6DOF::kPositionY)] << " "
                      << residVec[i][static_cast<int>(DataCSC_6DOF::kAngleX)] << " "
                      << residVec[i][static_cast<int>(DataCSC_6DOF::kAngleY)] << " " << "\n";
        }
        std::cout << "\n";
        std::cout << "-------------------------------------------------\n";
    }
    if (id.subdetId() == MuonSubdetId::DT)
    {
        DTChamberId dtId(id.rawId());
        std::cout << dtId << "\n";
        if (dtId.station() != 4)
        {
            for (int i = 0; i < nValues; ++i)
            {
                std::cout << residVec[i][static_cast<int>(DataDT_6DOF::kResidX)] << " "
                          << residVec[i][static_cast<int>(DataDT_6DOF::kResidY)] << " "
                          << residVec[i][static_cast<int>(DataDT_6DOF::kResSlopeX)] << " "
                          << residVec[i][static_cast<int>(DataDT_6DOF::kResSlopeY)] << " "
                          << residVec[i][static_cast<int>(DataDT_6DOF::kPositionX)] << " "
                          << residVec[i][static_cast<int>(DataDT_6DOF::kPositionY)] << " "
                          << residVec[i][static_cast<int>(DataDT_6DOF::kAngleX)] << " "
                          << residVec[i][static_cast<int>(DataDT_6DOF::kAngleY)] << " " << "\n";
            }
        }
        else
        {
            for (int i = 0; i < nValues; ++i)
            {
                std::cout << residVec[i][static_cast<int>(DataDT_5DOF::kResid)] << " "
                          << residVec[i][static_cast<int>(DataDT_5DOF::kResSlope)] << " "
                          << residVec[i][static_cast<int>(DataDT_5DOF::kPositionX)] << " "
                          << residVec[i][static_cast<int>(DataDT_5DOF::kPositionY)] << " "
                          << residVec[i][static_cast<int>(DataDT_5DOF::kAngleX)] << " "
                          << residVec[i][static_cast<int>(DataDT_5DOF::kAngleY)] << " " << "\n";
            }
        }
        std::cout << "\n";
        std::cout << "-------------------------------------------------\n";
    }
}

bool MuonResidualsGPRFitter::Select(DetId const& id) const
{
    if (m_options.empty()) return false;

    if (id.subdetId() == MuonSubdetId::CSC)
    {
        bool ringSelec, statSelec, endcSelec;
        CSCDetId cscId(id.rawId());
        int s = cscId.station();
        int e = cscId.endcap();
        int r = cscId.ring();
        endcSelec = m_options[Options::CSCEndcaps].empty() ? true : (m_options[Options::CSCEndcaps][e - 1] == '1');
        ringSelec = m_options[Options::CSCRings].empty() ? true : (m_options[Options::CSCRings][r - 1] == '1');
        statSelec = m_options[Options::CSCStations].empty() ? true : (m_options[Options::CSCStations][s - 1] == '1');
        return (endcSelec && ringSelec && statSelec); 
    }
    if (id.subdetId() == MuonSubdetId::DT)
    {
        DTChamberId dtId(id.rawId());
        int s = dtId.station();
        int w = dtId.wheel();
        // -2 <= w <= 2
        bool wheelSelec, statSelec;
        // if wheel string is empty, allow all wheels; same for stations
        wheelSelec = m_options[Options::DTWheels].empty() ? true : (m_options[Options::DTWheels][w + 2] == '1');
        statSelec = m_options[Options::DTStations].empty() ? true : (m_options[Options::DTStations][s - 1] == '1');
        return (wheelSelec && statSelec); 
    }
    return false;
}

void MuonResidualsGPRFitter::SetOptions(std::vector<std::string> const& options)
{
    if (options.size() != static_cast<size_t>(Options::OptCount))
    {
        std::cout << "Incompatible options, setting to default.\n";
        m_options.clear();
        return;
    }

    m_options = options;
}

void MuonResidualsGPRFitter::SetOption(size_t optId, std::string const& option)
{
    if (optId >= static_cast<size_t>(Options::OptCount)) return;
    if (m_options.empty()) m_options.resize(static_cast<size_t>(Options::OptCount));
    m_options[optId] = option;
}

void MuonResidualsGPRFitter::SaveResidDistr() const
{
    std::unique_ptr<TFile> tFile = std::make_unique<TFile>("residuals.root", "recreate");

    for (auto const& [id, resid_vec]: m_data)
    {
        if (id.subdetId() == MuonSubdetId::DT)
        {
            DTChamberId dtId(id.rawId());
            int wheel = dtId.wheel();
            int station = dtId.station();
            int sector = dtId.sector();

            char tree_name[20];
            std::sprintf(tree_name, "%d:%d:%d_tree", wheel, station, sector);
            std::unique_ptr<TTree> tTree = std::make_unique<TTree>(tree_name, tree_name);

            if (station != 4)
            {
                Double_t residX, residY, residSlopeX, residSlopeY;
                tTree->Branch("residX", &residX, "residX/D");
                tTree->Branch("residY", &residY, "residY/D");
                tTree->Branch("residSlopeX", &residSlopeX, "residSlopeX/D");
                tTree->Branch("residSlopeY", &residSlopeY, "residSlopeY/D");

                for (auto it = resid_vec.cbegin(); it != resid_vec.cend(); ++it)
                {
                    residX = (*it)[static_cast<int>(DT_6DOF::kResidX)];
                    residY = (*it)[static_cast<int>(DT_6DOF::kResidY)];
                    residSlopeX = (*it)[static_cast<int>(DT_6DOF::kResSlopeX)];
                    residSlopeY = (*it)[static_cast<int>(DT_6DOF::kResSlopeY)];
                    tTree->Fill();
                }

                tFile->Write();
            }
            else
            {
                Double_t resid, residSlope;
                tTree->Branch("resid", &resid, "resid/D");
                tTree->Branch("residSlope", &residSlope, "residSlope/D");

                for (auto it = resid_vec.cbegin(); it != resid_vec.cend(); ++it)
                {
                    resid = (*it)[static_cast<int>(DT_5DOF::kResid)];
                    residSlope = (*it)[static_cast<int>(DT_5DOF::kResSlope)];
                    tTree->Fill();
                }

                tFile->Write();
            }
        }

        if (id.subdetId() == MuonSubdetId::CSC)
        {
            std::cout << "saving CSC residuals\n";
        }
    }

    tFile->Close();
}

void MuonResidualsGPRFitter::SaveResidPeakDistr(std::vector<double> const& params) const
{
    std::unique_ptr<TFile> tFile = std::make_unique<TFile>("resid_peaks.root", "recreate");
    for (auto const& [id, resid_vec]: m_data)
    {
        if (id.subdetId() == MuonSubdetId::DT)
        {
            Surface::PositionType position = m_DTGeometry->idToDet(id)->position();
            Surface::RotationType orientation = m_DTGeometry->idToDet(id)->rotation();

            align::RotationType glob_rot = MatrixXYZ(params[3], params[4], params[5]); // proper global gpr matrix in Z(3)Y(2)X(1) convention; 
            align::RotationType loc_rot = orientation*glob_rot*orientation.transposed();
            align::EulerAngles locAngles = AnglesXYZ(loc_rot);

            double alignphix_l = locAngles[0];
            double alignphiy_l = locAngles[1];
            double alignphiz_l = locAngles[2]; 

            double dx, dy, dz;
            dx = glob_rot.xx()*position.x() + glob_rot.xy()*position.y() + glob_rot.xz()*position.z() - position.x();
            dy = glob_rot.yx()*position.x() + glob_rot.yy()*position.y() + glob_rot.yz()*position.z() - position.y();
            dz = glob_rot.zx()*position.x() + glob_rot.zy()*position.y() + glob_rot.zz()*position.z() - position.z();

            double alignx_g = params[0] + dx;
            double aligny_g = params[1] + dy;
            double alignz_g = params[2] + dz;

            GlobalVector translation_g = GlobalVector(alignx_g, aligny_g, alignz_g);
            LocalVector translation_l = m_DTGeometry->idToDet(id)->toLocal(translation_g);

            double alignx_l = translation_l.x(); 
            double aligny_l = translation_l.y(); 
            double alignz_l = translation_l.z(); 

            DTChamberId dtId(id.rawId());
            int wheel = dtId.wheel();
            int station = dtId.station();
            int sector = dtId.sector();

            char tree_name[20];
            std::sprintf(tree_name, "%d:%d:%d_tree", wheel, station, sector);
            std::unique_ptr<TTree> tTree = std::make_unique<TTree>(tree_name, tree_name);

            if (station != 4)
            {
                double residXpeak, residYpeak, residSlopeXpeak, residSlopeYpeak;
                tTree->Branch("residXpeak", &residXpeak, "residXpeak/D");
                tTree->Branch("residYpeak", &residYpeak, "residYpeak/D");
                tTree->Branch("residSlopeXpeak", &residSlopeXpeak, "residSlopeXpeak/D");
                tTree->Branch("residSlopeYpeak", &residSlopeYpeak, "residSlopeYpeak/D");

                for (auto it = resid_vec.cbegin(); it != resid_vec.cend(); ++it)
                {
                    double resslopeX = (*it)[static_cast<int>(DT_6DOF::kResSlopeX)];
                    double resslopeY = (*it)[static_cast<int>(DT_6DOF::kResSlopeY)];
                    double positionX = (*it)[static_cast<int>(DT_6DOF::kPositionX)];
                    double positionY = (*it)[static_cast<int>(DT_6DOF::kPositionY)];
                    double angleX = (*it)[static_cast<int>(DT_6DOF::kAngleX)];
                    double angleY = (*it)[static_cast<int>(DT_6DOF::kAngleY)];

                    residXpeak = ResidX({alignx_l, aligny_l, alignz_l, alignphix_l, alignphiy_l, alignphiz_l, positionX, positionY, angleX, angleY, resslopeX});
                    residYpeak = ResidY({alignx_l, aligny_l, alignz_l, alignphix_l, alignphiy_l, alignphiz_l, positionX, positionY, angleX, angleY, resslopeY});
                    residSlopeXpeak = ResidDXDZ({alignx_l, aligny_l, alignz_l, alignphix_l, alignphiy_l, alignphiz_l, positionX, positionY, angleX, angleY});
                    residSlopeYpeak = ResidDYDZ({alignx_l, aligny_l, alignz_l, alignphix_l, alignphiy_l, alignphiz_l, positionX, positionY, angleX, angleY});

                    tTree->Fill();
                }

                tFile->Write();
            }
            else
            {
                std::cout << "saving DT station 4 residual peaks\n";
                // Double_t resid, residSlope;
                // tTree->Branch("resid", &resid, "resid/D");
                // tTree->Branch("residSlope", &residSlope, "residSlope/D");

                // for (auto it = resid_vec.cbegin(); it != resid_vec.cend(); ++it)
                // {
                //     resid = (*it)[static_cast<int>(DT_5DOF::kResid)];
                //     residSlope = (*it)[static_cast<int>(DT_5DOF::kResSlope)];
                //     tTree->Fill();
                // }

                // tFile->Write();
            }
        }

        if (id.subdetId() == MuonSubdetId::CSC)
        {
            std::cout << "saving CSC residual peaks\n";
        }
    }

    tFile->Close();
}

void MuonResidualsGPRFitter::PlotContours(std::string selection, int n_points)
{
    if (selection.empty() || selection.size() > 4) return;

    MuonResidualsGPRFitterFitInfo* fitinfo = new MuonResidualsGPRFitterFitInfo(this);
    TMinuit* MuonResidualsGPRFitter_TMinuit = new TMinuit(npar());
    MuonResidualsGPRFitter_TMinuit->SetPrintLevel(-1);
    MuonResidualsGPRFitter_TMinuit->SetObjectFit(fitinfo);
    MuonResidualsGPRFitter_TMinuit->SetFCN(GlobalFitter_FCN);
    inform(MuonResidualsGPRFitter_TMinuit);

    int ndim = npar();
    std::vector<char const*> par_names{"dx", "dy", "dz", "dphix", "dphiy", "dphiz"};
    std::map<char, double> error_table{{'1', 3.615}, {'2', 5.325}, {'3', 6.295}, {'4', 8.405}};
    std::map<char, int> color_table{{'1', 2}, {'2', 3}, {'3', 4}, {'4', 6}};
    for (int i = 0; i < ndim; ++i)
    {
        MuonResidualsGPRFitter_TMinuit->DefineParameter(i, par_names[i], 0.0, 0.000001, 0.0, 0.0);
    }

    std::unique_ptr<TFile> file = std::make_unique<TFile>("contour_plots.root", "RECREATE");
    std::unique_ptr<TStyle> gStyle = std::make_unique<TStyle>();

    for (size_t i = 0; i < selection.size(); ++i)
    {
        double err_def = error_table[selection[i]];
        if (err_def == 0) continue;
        MuonResidualsGPRFitter_TMinuit->SetErrorDef(err_def*err_def);
        int color = color_table[selection[i]];

        for (int p1 = 0; p1 < ndim; ++p1)
        {
            for (int p2 = p1 + 1; p2 < ndim; ++p2)
            {
                std::unique_ptr<TCanvas> canvas = std::make_unique<TCanvas>("c", "c", 600, 450);
                TGraph* g = static_cast<TGraph*>(MuonResidualsGPRFitter_TMinuit->Contour(n_points, p1, p2));
                if (!g) 
                {
                    Tracer::instance() << "Null pointer returned when drawing " << err_def << " countour for " << par_names[p1] << " vs " << par_names[p2] << "\n";
                    continue;
                }
                try 
                {
                    g->GetXaxis()->SetTitle(par_names[p1]);
                    g->GetYaxis()->SetTitle(par_names[p2]);
                    g->SetLineColor(color);
                    g->SetTitle(Form("Parameter contour %s vs %s", par_names[p1], par_names[p2]));
                    g->Draw();
                    file->WriteObject(g, Form("%dsig_cnt_%s_%s", color, par_names[p1], par_names[p2]));
                }
                catch (...)
                {
                    Tracer::instance() << "Exception caught when drawing " << err_def << " countour for " << par_names[p1] << " vs " << par_names[p2] << "\n";
                }
            }
        }
    }

    file->Close();
}

void MuonResidualsGPRFitter::PlotContour(PARAMS par1, PARAMS par2, int n_points)
{
    MuonResidualsGPRFitterFitInfo* fitinfo = new MuonResidualsGPRFitterFitInfo(this);
    TMinuit* MuonResidualsGPRFitter_TMinuit = new TMinuit(npar());
    MuonResidualsGPRFitter_TMinuit->SetPrintLevel(1);
    MuonResidualsGPRFitter_TMinuit->SetObjectFit(fitinfo);
    MuonResidualsGPRFitter_TMinuit->SetFCN(GlobalFitter_FCN);
    inform(MuonResidualsGPRFitter_TMinuit);

    int p1idx = static_cast<int>(par1);
    int p2idx = static_cast<int>(par2);

    int ndim = npar();
    std::vector<char const*> par_names{"dx", "dy", "dz", "dphix", "dphiy", "dphiz"};
    for (int i = 0; i < ndim; ++i)
    {
        MuonResidualsGPRFitter_TMinuit->DefineParameter(i, par_names[i], 0.0, 0.000001, 0.0, 0.0);
    }

    std::unique_ptr<TCanvas> canvas = std::make_unique<TCanvas>("c", "c", 600, 450);
    canvas->SetGrid();
    canvas->SetTickx();
    canvas->SetTicky();
    auto mg = std::make_unique<TMultiGraph>("mg", Form("Parameter contour %s vs %s", par_names[p1idx], par_names[p2idx]));

    MuonResidualsGPRFitter_TMinuit->SetErrorDef(28.355625); // 90 CL
    TGraph* g1 = static_cast<TGraph*>(MuonResidualsGPRFitter_TMinuit->Contour(n_points, p1idx, p2idx));
    g1->SetLineColor(kRed);
    g1->SetLineWidth(3);
    g1->SetTitle("1 sig");
    g1->SetDrawOption("AL");

    MuonResidualsGPRFitter_TMinuit->SetErrorDef(39.627025); // 95 CL
    TGraph* g2 = static_cast<TGraph*>(MuonResidualsGPRFitter_TMinuit->Contour(n_points, p1idx, p2idx));
    g2->SetLineColor(kGreen);
    g2->SetLineWidth(3);
    g2->SetTitle("2 sig");
    g2->SetDrawOption("AL");

    MuonResidualsGPRFitter_TMinuit->SetErrorDef(70.64402); // 99 CL
    TGraph* g3 = static_cast<TGraph*>(MuonResidualsGPRFitter_TMinuit->Contour(n_points, p1idx, p2idx));
    g3->SetLineColor(kBlue);
    g3->SetLineWidth(3);
    g3->SetTitle("3 sig");
    g3->SetDrawOption("AL");

    mg->Add(g1);
    mg->Add(g2);
    mg->Add(g3);
    mg->GetXaxis()->SetTitle(par_names[p1idx]);
    mg->GetYaxis()->SetTitle(par_names[p2idx]);
    mg->SetTitle(Form("Parameter contour %s vs %s", par_names[p1idx], par_names[p2idx]));
    mg->Draw("ALP");

    canvas->SaveAs(Form("likelihood_plot_%s_%s.pdf", par_names[p1idx], par_names[p2idx]));
}

void MuonResidualsGPRFitter::CalcStats()
{
    if (m_data.empty()) return;

    double val;

    for (auto const& [id, resid_vec]: m_data)
    {
        int n = resid_vec.size();

        if (id.subdetId() == MuonSubdetId::DT)
        {
            DTChamberId dtId(id.rawId());
            int station = dtId.station();

            if (station != 4)
            {
                double sum_sqr_dx = 0.0;
                double sum_dx = 0.0;

                double sum_sqr_dy = 0.0;
                double sum_dy = 0.0;

                double sum_sqr_dxdz = 0.0;
                double sum_dxdz = 0.0;

                double sum_sqr_dydz = 0.0;
                double sum_dydz = 0.0;

                for (auto res_ptr: resid_vec)
                {
                    val = res_ptr[static_cast<int>(DT_6DOF::kResidX)];
                    sum_sqr_dx += val*val;
                    sum_dx += val;

                    val = res_ptr[static_cast<int>(DT_6DOF::kResidY)];
                    sum_sqr_dy += val*val;
                    sum_dy += val;

                    val = res_ptr[static_cast<int>(DT_6DOF::kResSlopeX)];
                    sum_sqr_dxdz += val*val;
                    sum_dxdz += val;

                    val = res_ptr[static_cast<int>(DT_6DOF::kResSlopeY)];
                    sum_sqr_dydz += val*val;
                    sum_dydz += val;
                }

                double residX_mean = sum_dx/n;
                double residY_mean = sum_dy/n;
                double resslopeX_mean = sum_dxdz/n;
                double resslopeY_mean = sum_dydz/n;

                double residX_stddev = sqrt(sum_sqr_dx/n - sum_dx*sum_dx/(n*n));
                double residY_stddev = sqrt(sum_sqr_dy/n - sum_dy*sum_dy/(n*n));
                double resslopeX_stddev = sqrt(sum_sqr_dxdz/n - sum_dxdz*sum_dxdz/(n*n));
                double resslopeY_stddev = sqrt(sum_sqr_dydz/n - sum_dydz*sum_dydz/(n*n));

                std::vector<double> stds{residX_stddev, residY_stddev, resslopeX_stddev, resslopeY_stddev};
                std::vector<double> means{residX_mean, residY_mean, resslopeX_mean, resslopeY_mean};
                // std::cout << dtId.wheel() << "/" << station << "/" << dtId.sector() << ": \n\t";
                // std::copy(stds.begin(), stds.end(), std::ostream_iterator<double>(std::cout, " "));
                // std::cout << "\n\t";
                // std::copy(means.begin(), means.end(), std::ostream_iterator<double>(std::cout, " "));
                // std::cout << "\n";
                // m_stddevs.insert({id, std::move(stds)});
                m_stats.insert({id, {std::move(means), std::move(stds)}});
            }
            else
            {
                double sum_sqr_dx = 0.0;
                double sum_dx = 0.0;

                double sum_sqr_dxdz = 0.0;
                double sum_dxdz = 0.0;

                for (auto res_ptr: resid_vec)
                {
                    val = res_ptr[static_cast<int>(DT_5DOF::kResid)];
                    sum_sqr_dx += val*val;
                    sum_dx += val;

                    val = res_ptr[static_cast<int>(DT_5DOF::kResSlope)];
                    sum_sqr_dxdz += val*val;
                    sum_dxdz += val;
                }

                double resid_mean = sum_dx/n;
                double resslope_mean = sum_dxdz/n;

                double resid_stddev = sqrt(sum_sqr_dx/n - sum_dx*sum_dx/(n*n));
                double resslope_stddev = sqrt(sum_sqr_dxdz/n - sum_dxdz*sum_dxdz/(n*n));

                std::vector<double> stds{resid_stddev, resslope_stddev};
                std::vector<double> means{resid_mean, resslope_mean};
                // m_stddevs.insert({id, std::move(stds)});
                m_stats.insert({id, {std::move(means), std::move(stds)}});
            }
        }

        if (id.subdetId() == MuonSubdetId::CSC)
        {
            double sum_sqr_dx = 0.0;
            double sum_dx = 0.0;

            double sum_sqr_dxdz = 0.0;
            double sum_dxdz = 0.0;

            for (auto res_ptr: resid_vec)
            {
                val = res_ptr[static_cast<int>(CSC_6DOF::kResid)];
                sum_sqr_dx += val*val;
                sum_dx += val;

                val = res_ptr[static_cast<int>(CSC_6DOF::kResSlope)];
                sum_sqr_dxdz += val*val;
                sum_dxdz += val;
            }

            double resid_mean = sum_dx/n;
            double resslope_mean = sum_dxdz/n;

            double resid_stddev = sqrt(sum_sqr_dx/n - sum_dx*sum_dx/(n*n));
            double resslope_stddev = sqrt(sum_sqr_dxdz/n - sum_dxdz*sum_dxdz/(n*n));

            std::vector<double> stds{resid_stddev, resslope_stddev};
            std::vector<double> means{resid_mean, resslope_mean};
            // m_stddevs.insert({id, std::move(stds)});
            m_stats.insert({id, {std::move(means), std::move(stds)}});
        }
    }
}

void MuonResidualsGPRFitter::Print(int nValues) const
{
    for (auto const& [id, resid_vec]: m_data)
    {
        Print(nValues, id);
    }
}