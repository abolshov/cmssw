#ifdef STANDALONE_FITTER
#include "MuonResidualsGPRFitter.h"
#else
#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsGPRFitter.h"
#endif
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include <fstream>
#include <set>
#include "TMath.h"
#include "TH1.h"
#include "TF1.h"
#include "TVector2.h"
#include "TRobustEstimator.h"
#include "Math/MinimizerOptions.h"
#include <fstream>
#include <sstream>
#include <map>

static TMinuit *MuonResidualsGPRFitter_TMinuit;

static int iterate_calls = 0;
static int FCN_calls = 0;
static int dofit_calls = 0;

using D_6DOF = MuonResidualsGPRFitter::Data_6DOF;
using D_5DOF = MuonResidualsGPRFitter::Data_5DOF;
using PARAMS = MuonResidualsGPRFitter::PARAMS;

namespace {
    TMinuit *minuit;

    double residual_x(double delta_x,
                      double delta_y,
                      double delta_z,
                      double delta_phix,
                      double delta_phiy,
                      double delta_phiz,
                      double track_x,
                      double track_y,
                      double track_dxdz,
                      double track_dydz,
                      double alphax,
                      double residual_dxdz) {
           return delta_x - track_dxdz * delta_z - track_y * track_dxdz * delta_phix + track_x * track_dxdz * delta_phiy -
           track_y * delta_phiz + residual_dxdz * alphax;
    }

    double residual_y(double delta_x,
                      double delta_y,
                      double delta_z,
                      double delta_phix,
                      double delta_phiy,
                      double delta_phiz,
                      double track_x,
                      double track_y,
                      double track_dxdz,
                      double track_dydz,
                      double alphay,
                      double residual_dydz) {
           return delta_y - track_dydz * delta_z - track_y * track_dydz * delta_phix + track_x * track_dydz * delta_phiy +
           track_x * delta_phiz + residual_dydz * alphay;
    }

    double residual_dxdz(double delta_x,
                         double delta_y,
                         double delta_z,
                         double delta_phix,
                         double delta_phiy,
                         double delta_phiz,
                         double track_x,
                         double track_y,
                         double track_dxdz,
                         double track_dydz) {
           return -track_dxdz * track_dydz * delta_phix + (1. + track_dxdz * track_dxdz) * delta_phiy -
           track_dydz * delta_phiz;
    }

    double residual_dydz(double delta_x,
                         double delta_y,
                         double delta_z,
                         double delta_phix,
                         double delta_phiy,
                         double delta_phiz,
                         double track_x,
                         double track_y,
                         double track_dxdz,
                         double track_dydz) {
           return -(1. + track_dydz * track_dydz) * delta_phix + track_dxdz * track_dydz * delta_phiy +
           track_dxdz * delta_phiz;
    }
    
    // function iterating over array of residuals IN SPECIFIC CHAMBER and calculating likelihood 
    // Notes:
    // - function takes iterator to where residuals for a chamber start and end; 
    //   I have to loop over chambers and call iterate twice to iterate over positive and negative residuals separately; 
    // - ...gamma - are not needed for log pure gaussian likelihood; I'm only plannng to use that one;
    // - alpha accounts for (residual - slope) corellation; set to zero in log pure gaussian; 
    // - allow only 6 parameter fit?
    // - pass dt geometry to do parameter conversion from global to local
    // - pass number of parameters used in FCN
}

double MuonResidualsGPRFitter_logPureGaussian(double residual, double center, double sigma) 
{
    sigma = fabs(sigma);
    static const double cgaus = 0.5 * log(2. * M_PI);
    return (-pow(residual - center, 2) * 0.5 / sigma / sigma) - cgaus - log(sigma);
}

MuonResidualsGPRFitter::MuonResidualsGPRFitter(const DTGeometry* dtGeometry)
    : m_gpr_dtGeometry(dtGeometry),
      m_printLevel(0),
      m_strategy(0),
      /* m_cov(1), */
      m_loglikelihood(0.0) {}

void MuonResidualsGPRFitter::fill(std::map<Alignable*, MuonResidualsTwoBin*>::const_iterator ali_and_data) 
{
    DetId id = ali_and_data->first->geomDetId();
    DTChamberId cham_id(id.rawId());
    if (id.subdetId() == MuonSubdetId::DT && cham_id.station() == 1) 
    {
        m_datamap.insert(*ali_and_data);
    }
}

// TO-DO: implement a function iterating over residuals in a chamber and calculating likelihood for that chamber 
// to avoid copying code in FCN; maybe should put it in the namespace

//FCN is fed to minuit; this is the function being minimized, i.e. log likelihood
void MuonResidualsGPRFitter_FCN(int &npar, double *gin, double &fval, double *par, int iflag) 
{
    MuonResidualsGPRFitterFitInfo *fitinfo = (MuonResidualsGPRFitterFitInfo *)(minuit->GetObjectFit());
    MuonResidualsGPRFitter *gpr_fitter = fitinfo->gpr_fitter();
    DTGeometry const* geom = gpr_fitter->getDTGeometry();

    fval = 0.0; // likelihood
    // loop over all chambers
    iterate_calls = 0;
    for (std::map<Alignable*, MuonResidualsTwoBin*>::const_iterator chamber_data = gpr_fitter->datamap_begin();
                                                                    chamber_data != gpr_fitter->datamap_end();
                                                                    ++chamber_data)
    {
        DetId id = chamber_data->first->geomDetId();

        DTChamberId cid(id.rawId());
        int station = cid.station();
        int wheel = cid.wheel();
        int sector = cid.sector();

        double alignx_g = par[static_cast<int>(PARAMS::kAlignX)];
        double aligny_g = par[static_cast<int>(PARAMS::kAlignY)];
        double alignz_g = par[static_cast<int>(PARAMS::kAlignZ)];
        double alignphix_g = par[static_cast<int>(PARAMS::kAlignPhiX)];
        double alignphiy_g = par[static_cast<int>(PARAMS::kAlignPhiY)];
        double alignphiz_g = par[static_cast<int>(PARAMS::kAlignPhiZ)];
        const double resXsigma = par[static_cast<int>(PARAMS::kResidXSigma)];
        const double resYsigma = par[static_cast<int>(PARAMS::kResidYSigma)];
        const double slopeXsigma = par[static_cast<int>(PARAMS::kResSlopeXSigma)];
        const double slopeYsigma = par[static_cast<int>(PARAMS::kResSlopeYSigma)];

        if (station == 1)
        {
            GlobalVector translation_g = GlobalVector(alignx_g, aligny_g, alignz_g);
            GlobalVector rotation_g = GlobalVector(alignphix_g, alignphiy_g, alignphiz_g);
            LocalVector translation_l = geom->idToDet(id)->toLocal(translation_g);
            LocalVector rotation_l = geom->idToDet(id)->toLocal(rotation_g);

            double const alignx_l = translation_l.x(); 
            double const aligny_l = translation_l.y(); 
            double const alignz_l = translation_l.z();
            double const alignphix_l = rotation_l.x();
            double const alignphiy_l = rotation_l.y();
            double const alignphiz_l = rotation_l.z(); 

            std::vector<double*>::const_iterator pos_begin = chamber_data->second->residualsPos_begin();
            std::vector<double*>::const_iterator pos_end = chamber_data->second->residualsPos_end();
            std::vector<double*>::const_iterator it = pos_begin;
            for (it = pos_begin; it != pos_end; ++it)
            {
                const double residX = (*it)[static_cast<int>(D_6DOF::kResidX)];
                const double residY = (*it)[static_cast<int>(D_6DOF::kResidY)];
                const double resslopeX = (*it)[static_cast<int>(D_6DOF::kResSlopeX)];
                const double resslopeY = (*it)[static_cast<int>(D_6DOF::kResSlopeY)];
                const double positionX = (*it)[static_cast<int>(D_6DOF::kPositionX)];
                const double positionY = (*it)[static_cast<int>(D_6DOF::kPositionY)];
                const double angleX = (*it)[static_cast<int>(D_6DOF::kAngleX)];
                const double angleY = (*it)[static_cast<int>(D_6DOF::kAngleY)];

                double alphaX = 0.0;
                double alphaY = 0.0;
                double residXpeak = residual_x(alignx_l, aligny_l, alignz_l, alignphix_l, alignphiy_l, alignphiz_l, positionX, positionY, angleX, angleY, alphaX, resslopeX);
                double residYpeak = residual_y(alignx_l, aligny_l, alignz_l, alignphix_l, alignphiy_l, alignphiz_l, positionX, positionY, angleX, angleY, alphaY, resslopeY);
                double slopeXpeak = residual_dxdz(alignx_l, aligny_l, alignz_l, alignphix_l, alignphiy_l, alignphiz_l, positionX, positionY, angleX, angleY);
                double slopeYpeak = residual_dydz(alignx_l, aligny_l, alignz_l, alignphix_l, alignphiy_l, alignphiz_l, positionX, positionY, angleX, angleY);

                double weight = 1.0;

                fval += -weight * MuonResidualsGPRFitter_logPureGaussian(residX, residXpeak, resXsigma);
                fval += -weight * MuonResidualsGPRFitter_logPureGaussian(residY, residYpeak, resYsigma);
                fval += -weight * MuonResidualsGPRFitter_logPureGaussian(resslopeX, slopeXpeak, slopeXsigma);
                fval += -weight * MuonResidualsGPRFitter_logPureGaussian(resslopeY, slopeYpeak, slopeYsigma);

                float inc = -weight*(MuonResidualsGPRFitter_logPureGaussian(residX, residXpeak, resXsigma)+
                                     MuonResidualsGPRFitter_logPureGaussian(residY, residYpeak, resYsigma)+
                                     MuonResidualsGPRFitter_logPureGaussian(resslopeX, slopeXpeak, slopeXsigma)+
                                     MuonResidualsGPRFitter_logPureGaussian(resslopeY, slopeYpeak, slopeYsigma));

                
                if ((abs(MuonResidualsGPRFitter_logPureGaussian(residX, residXpeak, resXsigma)) > 150 ||
                     abs(MuonResidualsGPRFitter_logPureGaussian(residY, residYpeak, resYsigma)) > 300 ||
                     abs(MuonResidualsGPRFitter_logPureGaussian(resslopeX, slopeXpeak, slopeXsigma)) > 1500 ||
                     abs(MuonResidualsGPRFitter_logPureGaussian(resslopeY, slopeYpeak, slopeYsigma)) > 3000)
                     && fval > 0.0)
                {
                    // std::cout << "===============================================" << std::endl;
                    std::cout << "Anomaly in " << wheel << "/" << station << "/" << sector << " detected:" << std::endl;
                    std::cout << "Initial likelihood = " << fval << std::endl;

                    std::cout << "Global to local transformation:" << std::endl;
                    std::cout << "----dx:    " << alignx_g << " ---> " << alignx_l << std::endl
                              << "----dy:    " << aligny_g << " ---> " << aligny_l << std::endl
                              << "----dz:    " << alignz_g << " ---> " << alignz_l << std::endl
                              << "----dphix: " << alignphix_g << " ---> " << alignphix_l << std::endl
                              << "----dphiy: " << alignphiy_g << " ---> " << alignphiy_l << std::endl
                              << "----dphiz: " << alignphiz_g << " ---> " << alignphiz_l << std::endl
                              << std::endl;

                    std::cout << "Parameter uncertainties:" << std::endl;          
                    std::cout << "----resXsigma = " << resXsigma << std::endl
                              << "----resYsigma = " << resYsigma << std::endl
                              << "----slopeXsigma = " << slopeXsigma << std::endl
                              << "----slopeYsigma = " << slopeYsigma << std::endl
                              << std::endl;

                    std::cout << "Residuals and coordinates:" << std::endl;
                    std::cout << "----residX = " << residX << std::endl;
                    std::cout << "----residY = " << residY << std::endl;
                    std::cout << "----resslopeX = " << resslopeX << std::endl;
                    std::cout << "----resslopeY = " << resslopeY << std::endl;
                    std::cout << "----positionX = " << positionX << std::endl;
                    std::cout << "----positionY = " << positionY << std::endl;
                    std::cout << "----angleX = " << angleX << std::endl;
                    std::cout << "----angleY = " << angleY << std::endl;
                    std::cout << std::endl;

                    std::cout << "Residuals peaks:" << std::endl;
                    std::cout << "----residXpeak = " << residXpeak << std::endl;
                    std::cout << "----residYpeak = " << residYpeak << std::endl;
                    std::cout << "----slopeXpeak = " << slopeXpeak << std::endl;
                    std::cout << "----slopeYpeak = " << slopeYpeak << std::endl;
                    std::cout << std::endl;

                    std::cout << "----logPureGaussian(residX, residXpeak, resXsigma) = " << MuonResidualsGPRFitter_logPureGaussian(residX, residXpeak, resXsigma) << std::endl;
                    std::cout << "----logPureGaussian(residY, residYpeak, resYsigma) = " << MuonResidualsGPRFitter_logPureGaussian(residY, residYpeak, resYsigma) << std::endl;
                    std::cout << "----logPureGaussian(resslopeX, slopeXpeak, slopeXsigma) = " << MuonResidualsGPRFitter_logPureGaussian(resslopeX, slopeXpeak, slopeXsigma) << std::endl;
                    std::cout << "----logPureGaussian(resslopeY, slopeYpeak, slopeYsigma) = " << MuonResidualsGPRFitter_logPureGaussian(resslopeY, slopeYpeak, slopeYsigma) << std::endl;

                    std::cout << "Likelihood increment:" << std::endl;
                    std::cout << "----inc = " << inc << std::endl;
                }
            }
        }
        else 
        {
            continue;
        }
        
    } // loop over chambers in the map ends
    std::cout << "******************************************" << std::endl;
    std::cout << "FCN call #: " << FCN_calls << std::endl;
    std::cout << "FCN = " << fval << std::endl;
    std::cout << "params: ";
    for (int i = 0; i < npar; ++i)
    {
        std::cout << par[i] << " ";
    } 
    std::cout << std::endl;
    std::cout << "******************************************" << std::endl;
    ++FCN_calls;
}

void MuonResidualsGPRFitter::inform(TMinuit *tMinuit) { minuit = tMinuit; }

bool MuonResidualsGPRFitter::dofit(void (*fcn)(int &, double *, double &, double *, int),
                                std::vector<int> &parNum,
                                std::vector<std::string> &parName,
                                std::vector<double> &start,
                                std::vector<double> &step,
                                std::vector<double> &low,
                                std::vector<double> &high)
{
    ++dofit_calls;
    MuonResidualsGPRFitterFitInfo *fitinfo = new MuonResidualsGPRFitterFitInfo(this);

    // define Minuit object
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
        // if (*iNum == static_cast<int>(PARAMS::kResSlopeYSigma))
        if (*iNum > 5)
        {
            MuonResidualsGPRFitter_TMinuit->FixParameter(*iNum);
        }
    }

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
    arglist[0] = 0.5;
    ierflg = 0;
    smierflg = 0;
    MuonResidualsGPRFitter_TMinuit->mnexcm("SET ERR", arglist, 1, ierflg);
    if (ierflg != 0) 
    {
        std::cout << "Failing after: SET ERR; ierflg != 0" << std::endl;
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
        std::cout << "Failing after: SET STR; ierflg != 0" << std::endl;
        delete MuonResidualsGPRFitter_TMinuit;
        delete fitinfo;
        return false;
    }

    double eps_machine(std::numeric_limits<double>::epsilon());
    for (int i = 0; i < 10; i++)
    {
        arglist[i] = 0.0;
    }
    arglist[0] = eps_machine;
    ierflg = 0;
    MuonResidualsGPRFitter_TMinuit->mnexcm("SET EPS", arglist, 1, ierflg);
    if (ierflg != 0) 
    {
        std::cout << "Failing after: SET EPS; ierflg != 0" << std::endl;
        delete MuonResidualsGPRFitter_TMinuit;
        delete fitinfo;
        return false;
    }

    // bool try_again = false;

    // minimize
    for (int i = 0; i < 10; i++)
    {
        arglist[i] = 0.0;
    }
    arglist[0] = 50000;
    ierflg = 0;
    MuonResidualsGPRFitter_TMinuit->mnexcm("MIGRAD", arglist, 1, ierflg);
    // log_file << "After: MIGRAD; ierflg = " << ierflg << std::endl;
    std::cout << "After: MIGRAD; ierflg = " << ierflg << std::endl;
    MuonResidualsGPRFitter_TMinuit->mnstat(fmin, fedm, errdef, npari, nparx, istat);
    std::cout << "Status code istat = " << istat << std::endl;
    std::cout << "Estimated distance to minimum fedm = " << fedm << std::endl;
    std::cout << "Best FCN value fmin = " << fmin << std::endl;
    // log_file << "Status code istat = " << istat << std::endl;
    // log_file << "Estimated distance to minimum fedm = " << fedm << std::endl;
    // log_file << "Best FCN value fmin = " << fmin << std::endl;
    // if (ierflg != 0)
    // {
    //     try_again = true;
    // }
    // just once more, if needed (using the final Minuit parameters from the failed fit; often works)
    // if (try_again) 
    // {
    //     for (int i = 0; i < 10; i++)
    //     {
    //         arglist[i] = 0.0;
    //     }
    //     arglist[0] = 50000;
    //     MuonResidualsGPRFitter_TMinuit->mnexcm("MIGRAD", arglist, 1, smierflg);
    //     // log_file << "After: MIGRAD in try_agaian; smierflg = " << smierflg << std::endl;
    //     std::cout << "After: MIGRAD in try_agaian; smierflg = " << smierflg << std::endl;
    //     MuonResidualsGPRFitter_TMinuit->mnstat(fmin, fedm, errdef, npari, nparx, istat);
    //     std::cout << "Status code istat = " << istat << std::endl;
    //     std::cout << "Estimated distance to minimum fedm = " << fedm << std::endl;
    //     std::cout << "Best FCN value fmin = " << fmin << std::endl;
    //     // log_file << "Status code istat = " << istat << std::endl;
    //     // log_file << "Estimated distance to minimum fedm = " << fedm << std::endl;
    //     // log_file << "Best FCN value fmin = " << fmin << std::endl;
    // }

    // MuonResidualsGPRFitter_TMinuit->mnstat(fmin, fedm, errdef, npari, nparx, istat);
    // std::cout << "After: mnstat; istat = " << istat << std::endl;
    // std::cout << "Estimated distance to minimum fedm = " << fedm << std::endl;
    // log_file << "After: mnstat; istat = " << istat << std::endl;
    // log_file << "Estimated distance to minimum fedm = " << fedm << std::endl;

    if (istat != 3) 
    {
        for (int i = 0; i < 10; i++)
        {
            arglist[i] = 0.0;
        }
        ierflg = 0;
        MuonResidualsGPRFitter_TMinuit->mnexcm("HESSE", arglist, 0, ierflg);
        MuonResidualsGPRFitter_TMinuit->mnstat(fmin, fedm, errdef, npari, nparx, istat);
        std::cout << "Status code istat = " << istat << std::endl;
        std::cout << "Estimated distance to minimum fedm = " << fedm << std::endl;
        std::cout << "Best FCN value fmin = " << fmin << std::endl;
        // log_file << "Status code istat = " << istat << std::endl;
        // log_file << "Estimated distance to minimum fedm = " << fedm << std::endl;
        // log_file << "Best FCN value fmin = " << fmin << std::endl;
        // MuonResidualsGPRFitter_TMinuit->mnexcm("MINOS", arglist, 0, ierflg);
    }

    // MuonResidualsGPRFitter_TMinuit->mnstat(fmin, fedm, errdef, npari, nparx, istat);
    // std::cout << "mnstat before exiting from dofit" << std::endl;
    // std::cout << "Status code istat = " << istat << std::endl;
    // std::cout << "Estimated distance to minimum fedm = " << fedm << std::endl;
    // std::cout << "Best FCN value fmin = " << fmin << std::endl;
    // log_file << "mnstat before exiting from dofit" << std::endl;
    // log_file << "Status code istat = " << istat << std::endl;
    // log_file << "Estimated distance to minimum fedm = " << fedm << std::endl;
    // log_file << "Best FCN value fmin = " << fmin << std::endl;

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
    // m_cov.ResizeTo(npar(), npar());
    // MuonResidualsGPRFitter_TMinuit->mnemat(m_cov.GetMatrixArray(), npar());

    // log_file << "FCN_calls = " << FCN_calls << std::endl;
    FCN_calls = 0;

    delete MuonResidualsGPRFitter_TMinuit;
    delete fitinfo;
    if (smierflg != 0)
    {
        std::cout << "After: HESSE; smierflg = " << smierflg << std::endl;
        // log_file << "After: HESSE; smierflg = " << smierflg << std::endl;
        return false;
    }    
    return true;

    // return true;
}

bool MuonResidualsGPRFitter::fit()
{
    double resx_std = 0.62;
    double resy_std = 0.95;
    double resslopex_std = 0.0033;
    double resslopey_std = 0.017;

    std::vector<int> nums{ static_cast<int>(PARAMS::kAlignX),
                           static_cast<int>(PARAMS::kAlignY),
                           static_cast<int>(PARAMS::kAlignZ),
                           static_cast<int>(PARAMS::kAlignPhiX),
                           static_cast<int>(PARAMS::kAlignPhiY),
                           static_cast<int>(PARAMS::kAlignPhiZ),
                           static_cast<int>(PARAMS::kResidXSigma),
                           static_cast<int>(PARAMS::kResidYSigma),
                           static_cast<int>(PARAMS::kResSlopeXSigma),
                           static_cast<int>(PARAMS::kResSlopeYSigma) };

    std::vector<std::string> names{ "AlignX",
                                    "AlignY",
                                    "AlignZ",
                                    "AlignPhiX",
                                    "AlignPhiY",
                                    "AlignPhiZ",
                                    "ResidXSigma",
                                    "ResidYSigma",
                                    "ResSlopeXSigma",
                                    "ResSlopeYSigma" };

    std::vector<double> starts{ 0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                0.0,
                                resx_std,
                                resy_std,
                                resslopex_std,
                                resslopey_std };

    std::vector<double> steps{ 0.01,
                               0.01,
                               0.01,
                               0.001,
                               0.001,
                               0.001,
                               0.001*resx_std,
                               0.001*resy_std,
                               0.001*resslopex_std,
                               0.001*resslopey_std };

    std::vector<double> lows{ 0.0, 0.0, 0.0, -0.1, -0.1, -0.1, 0.0, 0.0, 0.0, 0.0 };
    std::vector<double> highs{ 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 10.0, 10.0, 0.1, 0.1 };

    // std::random_device rd;
    // std::uniform_real_distribution<double> dist(-0.1, 0.1);
    // bool result = false;

    // std::ofstream log_file;
    // log_file.open("minimization.log");
    // log_file << "-----------------------------------------------------------------------" << std::endl;
    
    // for (size_t j = 0; j < 10; ++j)
    // {
    //     log_file << "Attempt #" << j + 1 << ":" << std::endl;
    //     std::cout << "Attempt #" << j + 1 << ":" << std::endl;
    //     std::vector<double> starts(6);
    //     for (size_t i = 0; i < 6; ++i)
    //     {
    //         starts[i] = dist(rd);
    //     }
    //     starts.push_back(resx_std);
    //     starts.push_back(resy_std);
    //     starts.push_back(resslopex_std);
    //     starts.push_back(resslopey_std);

    //     std::cout << "Starting point in parameter space:" << std::endl;
    //     for (size_t i = 0; i < 6; ++i)
    //     {
    //         std::cout << starts[i] << " ";
    //     }
    //     std::cout << std::endl;
    //     log_file << "Starting point in parameter space:" << std::endl;
    //     for (size_t i = 0; i < 6; ++i)
    //     {
    //         log_file << starts[i] << " ";
    //     }
    //     log_file << std::endl;
        
    //     result = dofit(&MuonResidualsGPRFitter_FCN, nums, names, starts, steps, lows, highs, log_file);
    //     if (result)
    //         break;
    //     log_file << "-----------------------------------------------------------------------" << std::endl;
    // }

    // log_file.close();

    // std::vector<double> starts(6);
    // for (size_t i = 0; i < 6; ++i)
    // {
    //     starts[i] = dist(rd);
    // }
    // starts.push_back(resx_std);
    // starts.push_back(resy_std);
    // starts.push_back(resslopex_std);
    // starts.push_back(resslopey_std);

    // std::cout << "Starting point in parameter space:" << std::endl;
    // for (size_t i = 0; i < 6; ++i)
    // {
    //     std::cout << starts[i] << " ";
    // }
    // std::cout << std::endl;

    // // result = dofit(&MuonResidualsGPRFitter_FCN, nums, names, starts, steps, lows, highs, log_file);
    // result = dofit(&MuonResidualsGPRFitter_FCN, nums, names, starts, steps, lows, highs);

    // return result;
    return dofit(&MuonResidualsGPRFitter_FCN, nums, names, starts, steps, lows, highs);
}


// ----------------------- old stuff -----------------------
/*
void MuonResidualsGPRFitter_FCN(int &npar, double *gin, double &fval, double *par, int iflag) 
{
    MuonResidualsGPRFitterFitInfo *fitinfo = (MuonResidualsGPRFitterFitInfo *)(minuit->GetObjectFit());
    MuonResidualsGPRFitter *gpr_fitter = fitinfo->gpr_fitter();
    DTGeometry const* gpr_DTGeometry = gpr_fitter->getDTGeometry();

    fval = 0.0; // likelihood
    // loop over all chambers
    iterate_calls = 0;
    for (std::map<Alignable*, MuonResidualsTwoBin*>::const_iterator chamber_data = gpr_fitter->datamap_begin();
                                                                    chamber_data != gpr_fitter->datamap_end();
                                                                    ++chamber_data)
    {
        // separating station 4 chambers (5 DOF) from 1,2,3 chambers (6 DOF) - done in iterate
        // chamber_data is pointing at current pair <Alignable*, MuonResidualsTwoBin*>
        // second contains ALL residuals of given chamber
        DetId id = chamber_data->first->geomDetId();

        DTChamberId cid(id.rawId());
        int station = cid.station();
        int wheel = cid.wheel();
        int sector = cid.sector();

        if (station == 1 && wheel == 1 && sector == 4)
        {
            // std::cout << "Chamber " << wheel << "/" << station << "/" << sector << " is being processed" << std::endl;

            // initialize iterators to arrays of pos and neg residuals
            std::vector<double*>::const_iterator pos_begin = chamber_data->second->residualsPos_begin();
            std::vector<double*>::const_iterator pos_end = chamber_data->second->residualsPos_end();
            // std::vector<double*>::const_iterator neg_begin = chamber_data->second->residualsNeg_begin();
            // std::vector<double*>::const_iterator neg_end = chamber_data->second->residualsNeg_end();

            // std::cout << "positive residuals:" << std::endl;
            iterate(fval, par, gpr_DTGeometry, id, pos_begin, pos_end);
            // std::cout << "negative residuals:" << std::endl;
            // iterate(fval, par, gpr_DTGeometry, id, neg_begin, neg_end);
        }
        else 
        {
            continue;
        }
        
    } // loop over chambers in the map ends
    std::cout << "******************************************" << std::endl;
    std::cout << "FCN call #: " << FCN_calls << std::endl;
    std::cout << "FCN = " << fval << std::endl;
    std::cout << "params: ";
    for (int i = 0; i < npar; ++i)
    {
        std::cout << par[i] << " ";
    } 
    std::cout << std::endl;
    std::cout << "******************************************" << std::endl;
    ++FCN_calls;
}

void iterate(double& fval, double* par, 
                 const DTGeometry* geom, DetId det_id,
                 std::vector<double*>::const_iterator begin,
                 std::vector<double*>::const_iterator end)
    {
        ++iterate_calls;
        // std::cout << "===============================================" << std::endl;
        DTChamberId curChamberId(det_id.rawId());
        int station = curChamberId.station();
        int wheel = curChamberId.wheel();
        int sector = curChamberId.sector();
        std::cout << "Chamber " << wheel << "/" << station << "/" << sector << " is being processed" << std::endl;
        std::cout << "iterate call # " << iterate_calls << std::endl;
        std::cout << "Likelihood passed fval = " << fval << std::endl;

        // Generate parameters:
        // _g postfix means quantity in global frame; _l - in local; no postfix - isn't needed to be converted
        double alignx_g = par[static_cast<int>(PARAMS::kAlignX)];
        double aligny_g = par[static_cast<int>(PARAMS::kAlignY)];
        double alignz_g = par[static_cast<int>(PARAMS::kAlignZ)];
        double alignphix_g = par[static_cast<int>(PARAMS::kAlignPhiX)];
        double alignphiy_g = par[static_cast<int>(PARAMS::kAlignPhiY)];
        double alignphiz_g = par[static_cast<int>(PARAMS::kAlignPhiZ)];
        const double resXsigma = par[static_cast<int>(PARAMS::kResidXSigma)];
        const double resYsigma = par[static_cast<int>(PARAMS::kResidYSigma)];
        const double slopeXsigma = par[static_cast<int>(PARAMS::kResSlopeXSigma)];
        const double slopeYsigma = par[static_cast<int>(PARAMS::kResSlopeYSigma)];
        // if I decide to take into account corellation between dx and dx/dz or dy and dy/dz need to add parameters alphaX and alphaY here
        // also need to add them in dofit and in TMinuit object
        // and of course add extra entries to PARAMS enum

        // std::cout << "Parameters in global frame:" << std::endl;
        // std::cout << "alignx_g = " << alignx_g << std::endl
        //           << "aligny_g = " << aligny_g << std::endl
        //           << "alignz_g = " << alignz_g << std::endl
        //           << "alignphix_g = " << alignphix_g << std::endl
        //           << "alignphiy_g = " << alignphiy_g << std::endl
        //           << "alignphiz_g = " << alignphiz_g << std::endl
        //           << "resXsigma = " << resXsigma << std::endl
        //           << "resYsigma = " << resYsigma << std::endl
        //           << "slopeXsigma = " << slopeXsigma << std::endl
        //           << "slopeYsigma = " << slopeYsigma << std::endl
        //           << std::endl;

        // Transform parameters to local
        GlobalVector translation_g = GlobalVector(alignx_g, aligny_g, alignz_g);
        GlobalVector rotation_g = GlobalVector(alignphix_g, alignphiy_g, alignphiz_g);
        LocalVector translation_l = geom->idToDet(det_id)->toLocal(translation_g);
        LocalVector rotation_l = geom->idToDet(det_id)->toLocal(rotation_g);

        // std::cout << "Parameters in local frame:" << std::endl;
        // std::cout << "alignx_l = " << translation_l.x() << std::endl
        //           << "aligny_l = " << translation_l.y() << std::endl
        //           << "alignz_l = " << translation_l.z() << std::endl
        //           << "alignphix_l = " << rotation_l.x() << std::endl
        //           << "alignphiy_l = " << rotation_l.y() << std::endl
        //           << "alignphiz_l = " << rotation_l.z() << std::endl
        //           << "resXsigma = " << resXsigma << std::endl
        //           << "resYsigma = " << resYsigma << std::endl
        //           << "slopeXsigma = " << slopeXsigma << std::endl
        //           << "slopeYsigma = " << slopeYsigma << std::endl
        //           << std::endl;

        double const alignx_l = translation_l.x(); 
        double const aligny_l = translation_l.y(); 
        double const alignz_l = translation_l.z();
        double const alignphix_l = rotation_l.x();
        double const alignphiy_l = rotation_l.y();
        double const alignphiz_l = rotation_l.z(); 

        // initialize contribution to likelihood from current alignable
        float fval_it = 0.0;

        if (station == 1 || station == 2 || station == 3)
        {
            // do 6 DOF iteration
            std::vector<double*>::const_iterator it = begin;
            for (it = begin; it != end; ++it)
            {
                const double residX = (*it)[static_cast<int>(D_6DOF::kResidX)];
                const double residY = (*it)[static_cast<int>(D_6DOF::kResidY)];
                const double resslopeX = (*it)[static_cast<int>(D_6DOF::kResSlopeX)];
                const double resslopeY = (*it)[static_cast<int>(D_6DOF::kResSlopeY)];
                const double positionX = (*it)[static_cast<int>(D_6DOF::kPositionX)];
                const double positionY = (*it)[static_cast<int>(D_6DOF::kPositionY)];
                const double angleX = (*it)[static_cast<int>(D_6DOF::kAngleX)];
                const double angleY = (*it)[static_cast<int>(D_6DOF::kAngleY)];
                // const double redchi2 = (*it)[MuonResidualsGPRFitter::kRedChi2];

                // std::cout << "residX = " << residX << std::endl;
                // std::cout << "residY = " << residY << std::endl;
                // std::cout << "resslopeX = " << resslopeX << std::endl;
                // std::cout << "resslopeY = " << resslopeY << std::endl;
                // std::cout << "positionX = " << positionX << std::endl;
                // std::cout << "positionY = " << positionY << std::endl;
                // std::cout << "angleX = " << angleX << std::endl;
                // std::cout << "angleY = " << angleY << std::endl;

                double alphaX = 0.0;
                double alphaY = 0.0;
                double residXpeak = residual_x(alignx_l, aligny_l, alignz_l, alignphix_l, alignphiy_l, alignphiz_l, positionX, positionY, angleX, angleY, alphaX, resslopeX);
                double residYpeak = residual_y(alignx_l, aligny_l, alignz_l, alignphix_l, alignphiy_l, alignphiz_l, positionX, positionY, angleX, angleY, alphaY, resslopeY);
                double slopeXpeak = residual_dxdz(alignx_l, aligny_l, alignz_l, alignphix_l, alignphiy_l, alignphiz_l, positionX, positionY, angleX, angleY);
                double slopeYpeak = residual_dydz(alignx_l, aligny_l, alignz_l, alignphix_l, alignphiy_l, alignphiz_l, positionX, positionY, angleX, angleY);

                // std::cout << "residXpeak = " << residXpeak << std::endl;
                // std::cout << "residYpeak = " << residYpeak << std::endl;
                // std::cout << "slopeXpeak = " << slopeXpeak << std::endl;
                // std::cout << "slopeYpeak = " << slopeYpeak << std::endl;

                // no weightened alignment yet; to make it possible need to add sumofweights() function to my class
                // should I sum weights of all hits for all chambers?
                // double weight = (1. / redchi2) * number_of_hits / sum_of_weights;
                // if (!weight_alignment) weight = 1.0;
                double weight = 1.0;

                // if (!weight_alignment || TMath::Prob(redchi2 * 12, 12) < 0.99)
                // {

                fval_it += -weight * MuonResidualsGPRFitter_logPureGaussian(residX, residXpeak, resXsigma);
                fval_it += -weight * MuonResidualsGPRFitter_logPureGaussian(residY, residYpeak, resYsigma);
                fval_it += -weight * MuonResidualsGPRFitter_logPureGaussian(resslopeX, slopeXpeak, slopeXsigma);
                fval_it += -weight * MuonResidualsGPRFitter_logPureGaussian(resslopeY, slopeYpeak, slopeYsigma);

                float inc = -weight*(MuonResidualsGPRFitter_logPureGaussian(residX, residXpeak, resXsigma)+
                                     MuonResidualsGPRFitter_logPureGaussian(residY, residYpeak, resYsigma)+
                                     MuonResidualsGPRFitter_logPureGaussian(resslopeX, slopeXpeak, slopeXsigma)+
                                     MuonResidualsGPRFitter_logPureGaussian(resslopeY, slopeYpeak, slopeYsigma));

                // if (residX > 1.5 || 
                //     residY > 3.0 || 
                //     resslopeX > 1.5 || 
                //     resslopeY > 3.0 ||
                //     residXpeak > 1.5 ||
                //     residYpeak > 1.5 ||
                //     slopeXpeak > 1.5 ||
                //     slopeYpeak > 1.5 ||
                //     (abs(alignx_l - alignx_g) > 1) ||
                //     (abs(aligny_l - aligny_g) > 1) ||
                //     (abs(alignz_l - alignz_g) > 1) ||
                //     (abs(alignphix_l - alignphix_g) > 1) ||
                //     (abs(alignphiy_l - alignphiy_g) > 1) ||
                //     (abs(alignphiz_l - alignphiz_g) > 1) || )
                if ((abs(MuonResidualsGPRFitter_logPureGaussian(residX, residXpeak, resXsigma)) > 150 ||
                     abs(MuonResidualsGPRFitter_logPureGaussian(residY, residYpeak, resYsigma)) > 300 ||
                     abs(MuonResidualsGPRFitter_logPureGaussian(resslopeX, slopeXpeak, slopeXsigma)) > 1500 ||
                     abs(MuonResidualsGPRFitter_logPureGaussian(resslopeY, slopeYpeak, slopeYsigma)) > 3000)
                     && fval_it > 0.0)
                {
                    // std::cout << "===============================================" << std::endl;
                    std::cout << "Anomaly in " << wheel << "/" << station << "/" << sector << " detected:" << std::endl;
                    std::cout << "Initial likelihood = " << fval_it << std::endl;

                    std::cout << "Global to local transformation:" << std::endl;
                    std::cout << "----dx:    " << alignx_g << " ---> " << alignx_l << std::endl
                              << "----dy:    " << aligny_g << " ---> " << aligny_l << std::endl
                              << "----dz:    " << alignz_g << " ---> " << alignz_l << std::endl
                              << "----dphix: " << alignphix_g << " ---> " << alignphix_l << std::endl
                              << "----dphiy: " << alignphiy_g << " ---> " << alignphiy_l << std::endl
                              << "----dphiz: " << alignphiz_g << " ---> " << alignphiz_l << std::endl
                              << std::endl;

                    std::cout << "Parameter uncertainties:" << std::endl;          
                    std::cout << "----resXsigma = " << resXsigma << std::endl
                              << "----resYsigma = " << resYsigma << std::endl
                              << "----slopeXsigma = " << slopeXsigma << std::endl
                              << "----slopeYsigma = " << slopeYsigma << std::endl
                              << std::endl;

                    std::cout << "Residuals and coordinates:" << std::endl;
                    std::cout << "----residX = " << residX << std::endl;
                    std::cout << "----residY = " << residY << std::endl;
                    std::cout << "----resslopeX = " << resslopeX << std::endl;
                    std::cout << "----resslopeY = " << resslopeY << std::endl;
                    std::cout << "----positionX = " << positionX << std::endl;
                    std::cout << "----positionY = " << positionY << std::endl;
                    std::cout << "----angleX = " << angleX << std::endl;
                    std::cout << "----angleY = " << angleY << std::endl;
                    std::cout << std::endl;

                    std::cout << "Residuals peaks:" << std::endl;
                    std::cout << "----residXpeak = " << residXpeak << std::endl;
                    std::cout << "----residYpeak = " << residYpeak << std::endl;
                    std::cout << "----slopeXpeak = " << slopeXpeak << std::endl;
                    std::cout << "----slopeYpeak = " << slopeYpeak << std::endl;
                    std::cout << std::endl;

                    std::cout << "----logPureGaussian(residX, residXpeak, resXsigma) = " << MuonResidualsGPRFitter_logPureGaussian(residX, residXpeak, resXsigma) << std::endl;
                    std::cout << "----logPureGaussian(residY, residYpeak, resYsigma) = " << MuonResidualsGPRFitter_logPureGaussian(residY, residYpeak, resYsigma) << std::endl;
                    std::cout << "----logPureGaussian(resslopeX, slopeXpeak, slopeXsigma) = " << MuonResidualsGPRFitter_logPureGaussian(resslopeX, slopeXpeak, slopeXsigma) << std::endl;
                    std::cout << "----logPureGaussian(resslopeY, slopeYpeak, slopeYsigma) = " << MuonResidualsGPRFitter_logPureGaussian(resslopeY, slopeYpeak, slopeYsigma) << std::endl;

                    std::cout << "Likelihood increment:" << std::endl;
                    std::cout << "----inc = " << inc << std::endl;


                    // std::cout << "===============================================" << std::endl;
                }
            }

            fval += fval_it;
            std::cout << "Likelihood incremented by fval_it = " << fval_it << std::endl;
            std::cout << "===============================================" << std::endl;
        }
        else if (station == 4)
        {
            // do 5 DOF iteration
            std::vector<double*>::const_iterator it = begin;
            for(it = begin; it != end; ++it)
            {
                const double residX = (*it)[static_cast<int>(D_5DOF::kResid)];
                const double resslopeX = (*it)[static_cast<int>(D_5DOF::kResSlope)];
                const double positionX = (*it)[static_cast<int>(D_5DOF::kPositionX)];
                const double positionY = (*it)[static_cast<int>(D_5DOF::kPositionY)];
                const double angleX = (*it)[static_cast<int>(D_5DOF::kAngleX)];
                const double angleY = (*it)[static_cast<int>(D_5DOF::kAngleY)];
                // const double redchi2 = (*it)[static_cast<int>(D_5DOF::kRedChi2)];

                double alphaX = 0.0;
                double residXpeak = residual_x(alignx_l, 0.0, alignz_l, alignphix_l, alignphiy_l, alignphiz_l, positionX, positionY, angleX, angleY, alphaX, resslopeX);
                double slopeXpeak = residual_dxdz(alignx_l, 0.0, alignz_l, alignphix_l, alignphiy_l, alignphiz_l, positionX, positionY, angleX, angleY);
                double weight = 1.0;
                fval_it += -weight * MuonResidualsFitter_logPureGaussian(residX, residXpeak, resXsigma);
                fval_it += -weight * MuonResidualsFitter_logPureGaussian(resslopeX, slopeXpeak, slopeXsigma);
            }

            fval += fval_it;
            // std::cout << "Likelihood incremented by fval_it = " << fval_it << std::endl;
            // std::cout << "===============================================" << std::endl;
        }
        else 
        {
            std::cout << "Error in iterate: wrong station" << std::endl;
        }
    }

}
*/