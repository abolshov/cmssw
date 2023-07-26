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
#include "TFile.h"
#include "TH2F.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TGraph2D.h"
#include "TRobustEstimator.h"
#include "Math/MinimizerOptions.h"
#include <sstream>
#include <map>
#include <functional>
#include <algorithm>
#include <iterator>
#include <chrono>
#include <cmath>

#include "Alignment/MuonAlignmentAlgorithms/interface/Tracer.hpp"
#include "Alignment/CommonAlignment/interface/Utilities.h"

#include "TMatrixDSym.h"

static TMinuit *MuonResidualsGPRFitter_TMinuit;

static double avg_call_duration = 0.0f;
static int iterate_calls = 0;
static int FCN_calls = 0;
static int dofit_calls = 0;

using D_6DOF = MuonResidualsGPRFitter::Data_6DOF;
using D_5DOF = MuonResidualsGPRFitter::Data_5DOF;
using PARAMS = MuonResidualsGPRFitter::PARAMS;
using ResidSigTypes =  MuonResidualsGPRFitter::ResidSigTypes;

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
}

double MuonResidualsGPRFitter_logPureGaussian(double residual, double center, double sigma) 
{
    sigma = fabs(sigma);
    static const double cgaus = 0.5 * log(2. * M_PI);
    return (-pow(residual - center, 2) * 0.5 / sigma / sigma) - cgaus - log(sigma);
}

MuonResidualsGPRFitter::MuonResidualsGPRFitter(DTGeometry const* dtGeometry,
                                               std::map<Alignable*, MuonResidualsTwoBin*> const& datamap,
                                               std::map<DetId, std::vector<double>> const& reswidths)
    : m_gpr_dtGeometry(dtGeometry),
      m_datamap(datamap),
      m_resWidths(reswidths),
      m_printLevel(0),
      m_strategy(0),
      m_value(npar(), 0.0),
      m_error(npar(), 0.0),
      m_loglikelihood(0.0) {}

void MuonResidualsGPRFitter::fill(std::map<Alignable*, MuonResidualsTwoBin*>::const_iterator it) 
{
    DetId id = it->first->geomDetId();
    DTChamberId cham_id(id.rawId());
    if (id.subdetId() == MuonSubdetId::DT && cham_id.station() == 1) 
    {
        m_datamap.insert(*it);
    }
}

//FCN is fed to minuit; this is the function being minimized, i.e. log likelihood
void MuonResidualsGPRFitter_FCN(int &npar, double *gin, double &fval, double *par, int iflag) 
{
    MuonResidualsGPRFitterFitInfo *fitinfo = (MuonResidualsGPRFitterFitInfo *)(minuit->GetObjectFit());
    MuonResidualsGPRFitter *gpr_fitter = fitinfo->gpr_fitter();
    DTGeometry const* geom = gpr_fitter->getDTGeometry();

    fval = 0.0; // likelihood
    // loop over all chambers
    iterate_calls = 0;

    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    for (std::map<Alignable*, MuonResidualsTwoBin*>::const_iterator chamber_data = gpr_fitter->datamap_begin();
                                                                    chamber_data != gpr_fitter->datamap_end();
                                                                    ++chamber_data)
    {
        DetId id = chamber_data->first->geomDetId();

        Alignable const* ali = chamber_data->first;
        align::PositionType const& position = ali->globalPosition();
        align::RotationType const& orientation = ali->globalRotation();

        DTChamberId cid(id.rawId());
        int station = cid.station();
        int wheel = cid.wheel();
        int sector = cid.sector();

        align::EulerAngles globAngles(3);
        globAngles[0] = par[static_cast<int>(PARAMS::kAlignPhiX)];
        globAngles[1] = par[static_cast<int>(PARAMS::kAlignPhiY)];
        globAngles[2] = par[static_cast<int>(PARAMS::kAlignPhiZ)];
        align::RotationType mtrx = align::toMatrix(globAngles);
        align::EulerAngles locAngles = align::toAngles(orientation*mtrx*orientation.transposed());

        double dx, dy, dz;
        dx = mtrx.xx()*position.x() + mtrx.yx()*position.y() + mtrx.zx()*position.z() - position.x();
        dy = mtrx.xy()*position.x() + mtrx.yy()*position.y() + mtrx.zy()*position.z() - position.y();
        dz = mtrx.xz()*position.x() + mtrx.yz()*position.y() + mtrx.zz()*position.z() - position.z();

        // double alignx_g = par[static_cast<int>(PARAMS::kAlignX)];
        // double aligny_g = par[static_cast<int>(PARAMS::kAlignY)];
        // double alignz_g = par[static_cast<int>(PARAMS::kAlignZ)];
        double alignx_g = par[static_cast<int>(PARAMS::kAlignX)] + dx;
        double aligny_g = par[static_cast<int>(PARAMS::kAlignY)] + dy;
        double alignz_g = par[static_cast<int>(PARAMS::kAlignZ)] + dz;
        // double alignphix_g = par[static_cast<int>(PARAMS::kAlignPhiX)];
        // double alignphiy_g = par[static_cast<int>(PARAMS::kAlignPhiY)];
        // double alignphiz_g = par[static_cast<int>(PARAMS::kAlignPhiZ)];

        std::vector<double> resWidths = gpr_fitter->getResWidths(id);
        double resXsigma = resWidths[static_cast<int>(ResidSigTypes::kResXSigma)];
        double resYsigma = resWidths[static_cast<int>(ResidSigTypes::kResYSigma)];
        double slopeXsigma = resWidths[static_cast<int>(ResidSigTypes::kResXslopeSigma)];
        double slopeYsigma = resWidths[static_cast<int>(ResidSigTypes::kResYslopeSigma)];

        GlobalVector translation_g = GlobalVector(alignx_g, aligny_g, alignz_g);
        // GlobalVector rotation_g = GlobalVector(alignphix_g, alignphiy_g, alignphiz_g);
        LocalVector translation_l = geom->idToDet(id)->toLocal(translation_g);
        // LocalVector rotation_l = geom->idToDet(id)->toLocal(rotation_g);

        double const alignx_l = translation_l.x(); 
        double const aligny_l = translation_l.y(); 
        double const alignz_l = translation_l.z();
        // double const alignphix_l = rotation_l.x();
        // double const alignphiy_l = rotation_l.y();
        // double const alignphiz_l = rotation_l.z(); 
        double const alignphix_l = locAngles[0];
        double const alignphiy_l = locAngles[1];
        double const alignphiz_l = locAngles[2]; 

        if (station != 4)
        {
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

                double inc = -weight*(MuonResidualsGPRFitter_logPureGaussian(residX, residXpeak, resXsigma)+
                                     MuonResidualsGPRFitter_logPureGaussian(residY, residYpeak, resYsigma)+
                                     MuonResidualsGPRFitter_logPureGaussian(resslopeX, slopeXpeak, slopeXsigma)+
                                     MuonResidualsGPRFitter_logPureGaussian(resslopeY, slopeYpeak, slopeYsigma));

                
                // if ((abs(MuonResidualsGPRFitter_logPureGaussian(residX, residXpeak, resXsigma)) > 150 ||
                //      abs(MuonResidualsGPRFitter_logPureGaussian(residY, residYpeak, resYsigma)) > 300 ||
                //      abs(MuonResidualsGPRFitter_logPureGaussian(resslopeX, slopeXpeak, slopeXsigma)) > 1500 ||
                //      abs(MuonResidualsGPRFitter_logPureGaussian(resslopeY, slopeYpeak, slopeYsigma)) > 3000)
                //      && fval > 0.0)
                // {
                //     // std::cout << "===============================================" << std::endl;
                //     std::cout << "Anomaly in " << wheel << "/" << station << "/" << sector << " detected:" << std::endl;
                //     std::cout << "Initial likelihood = " << fval << std::endl;

                //     std::cout << "Global to local transformation:" << std::endl;
                //     std::cout << "----dx:    " << alignx_g << " ---> " << alignx_l << std::endl
                //               << "----dy:    " << aligny_g << " ---> " << aligny_l << std::endl
                //               << "----dz:    " << alignz_g << " ---> " << alignz_l << std::endl
                //               << "----dphix: " << alignphix_g << " ---> " << alignphix_l << std::endl
                //               << "----dphiy: " << alignphiy_g << " ---> " << alignphiy_l << std::endl
                //               << "----dphiz: " << alignphiz_g << " ---> " << alignphiz_l << std::endl
                //               << std::endl;

                //     std::cout << "Parameter uncertainties:" << std::endl;          
                //     std::cout << "----resXsigma = " << resXsigma << std::endl
                //               << "----resYsigma = " << resYsigma << std::endl
                //               << "----slopeXsigma = " << slopeXsigma << std::endl
                //               << "----slopeYsigma = " << slopeYsigma << std::endl
                //               << std::endl;

                //     std::cout << "Residuals and coordinates:" << std::endl;
                //     std::cout << "----residX = " << residX << std::endl;
                //     std::cout << "----residY = " << residY << std::endl;
                //     std::cout << "----resslopeX = " << resslopeX << std::endl;
                //     std::cout << "----resslopeY = " << resslopeY << std::endl;
                //     std::cout << "----positionX = " << positionX << std::endl;
                //     std::cout << "----positionY = " << positionY << std::endl;
                //     std::cout << "----angleX = " << angleX << std::endl;
                //     std::cout << "----angleY = " << angleY << std::endl;
                //     std::cout << std::endl;

                //     std::cout << "Residuals peaks:" << std::endl;
                //     std::cout << "----residXpeak = " << residXpeak << std::endl;
                //     std::cout << "----residYpeak = " << residYpeak << std::endl;
                //     std::cout << "----slopeXpeak = " << slopeXpeak << std::endl;
                //     std::cout << "----slopeYpeak = " << slopeYpeak << std::endl;
                //     std::cout << std::endl;

                //     std::cout << "----logPureGaussian(residX, residXpeak, resXsigma) = " << MuonResidualsGPRFitter_logPureGaussian(residX, residXpeak, resXsigma) << std::endl;
                //     std::cout << "----logPureGaussian(residY, residYpeak, resYsigma) = " << MuonResidualsGPRFitter_logPureGaussian(residY, residYpeak, resYsigma) << std::endl;
                //     std::cout << "----logPureGaussian(resslopeX, slopeXpeak, slopeXsigma) = " << MuonResidualsGPRFitter_logPureGaussian(resslopeX, slopeXpeak, slopeXsigma) << std::endl;
                //     std::cout << "----logPureGaussian(resslopeY, slopeYpeak, slopeYsigma) = " << MuonResidualsGPRFitter_logPureGaussian(resslopeY, slopeYpeak, slopeYsigma) << std::endl;

                //     std::cout << "Likelihood increment:" << std::endl;
                //     std::cout << "----inc = " << inc << std::endl;
                // }
            }
        }
        else 
        {
            continue;
        }
        
    } // loop over chambers in the map ends
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    std::cout << "******************************************" << std::endl;
    std::cout << "FCN call #: " << FCN_calls << std::endl;
    std::cout << "execution time " << duration.count() << " ms" << std::endl;
    std::cout << "FCN = " << fval << std::endl;
    std::cout << "params: ";
    for (int i = 0; i < npar; ++i)
    {
        std::cout << par[i] << " ";
    } 
    std::cout << std::endl;
    std::cout << "******************************************" << std::endl;
    ++FCN_calls;
    avg_call_duration += static_cast<double>(duration.count());
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
    arglist[0] = 0.5;
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

    // double eps_machine(std::numeric_limits<double>::epsilon());
    double eps = 1e-10;
    for (int i = 0; i < 10; i++)
    {
        arglist[i] = 0.0;
    }
    arglist[0] = eps;
    ierflg = 0;
    MuonResidualsGPRFitter_TMinuit->mnexcm("SET EPS", arglist, 1, ierflg);
    if (ierflg != 0) 
    {
        delete MuonResidualsGPRFitter_TMinuit;
        delete fitinfo;
        return false;
    }

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

    double ratio = fedm/errdef;
    double tolerFactor = (ratio < 1.0) ? 5.0 : std::ceil(ratio);
    Tracer::instance() << "First MIGRAD:\n"
                       << "fmin = " << fmin << "\n"
                       << "fedm = " << fedm << "\n"
                       << "istat = " << istat << "\n"
                       << "ierflg = " << ierflg << "\n"
                       << "fedm/errdef = " << ratio << "\n"
                       << "tolerFactor = " << tolerFactor << "\n";

    size_t nPar = npar();
    std::vector<double> fMIGRAD_value(nPar, 0.0);
    std::vector<double> fMIGRAD_error(nPar, 0.0);
    for (size_t i = 0; i < nPar; ++i)
    {
        double v, e;
        MuonResidualsGPRFitter_TMinuit->GetParameter(i, v, e);
        fMIGRAD_value[i] = v;
        fMIGRAD_error[i] = e;
    }

    Tracer::instance() << "fMIGRAD_value: ";
    Tracer::instance().copy(fMIGRAD_value.begin(), fMIGRAD_value.end());
    Tracer::instance() << "\n";
    Tracer::instance() << "fMIGRAD_error: ";
    Tracer::instance().copy(fMIGRAD_error.begin(), fMIGRAD_error.end());
    Tracer::instance() << "\n";

    
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
        // arglist[1] = tolerFactor;
        arglist[1] = tolerFactor*1000;
        smierflg = 0;
        MuonResidualsGPRFitter_TMinuit->mnexcm("MIGRAD", arglist, 2, smierflg);
        // MuonResidualsGPRFitter_TMinuit->mnexcm("SIMPLEX", arglist, 2, smierflg);
        MuonResidualsGPRFitter_TMinuit->mnstat(fmin, fedm, errdef, npari, nparx, istat);

        Tracer::instance() << "Second MIGRAD:\n"
                           << "fmin = " << fmin << "\n"
                           << "fedm = " << fedm << "\n"
                           << "istat = " << istat << "\n"
                           << "smierflg = " << smierflg << "\n"; 
    }

    std::vector<double> sMIGRAD_value(nPar, 0.0);
    std::vector<double> sMIGRAD_error(nPar, 0.0);
    if (try_again)
    {
        for (size_t i = 0; i < nPar; ++i)
        {
            double v, e;
            MuonResidualsGPRFitter_TMinuit->GetParameter(i, v, e);
            sMIGRAD_value[i] = v;
            sMIGRAD_error[i] = e;
        }
        Tracer::instance() << "sMIGRAD_value: ";
        Tracer::instance().copy(sMIGRAD_value.begin(), sMIGRAD_value.end());
        Tracer::instance() << "\n";
        Tracer::instance() << "sMIGRAD_error: ";
        Tracer::instance().copy(sMIGRAD_error.begin(), sMIGRAD_error.end());
        Tracer::instance() << "\n";
    }

    int sierflg = 0; // SIMPLEX error flag
    if (try_again && smierflg != 0)
    {
        for (int i = 0; i < 10; i++)
        {
            arglist[i] = 0.0;
        }
        arglist[0] = 50000;
        arglist[1] = tolerFactor;
        MuonResidualsGPRFitter_TMinuit->mnexcm("SIMPLEX", arglist, 2, sierflg);
        MuonResidualsGPRFitter_TMinuit->mnstat(fmin, fedm, errdef, npari, nparx, istat);
        Tracer::instance() << "SIMPLEX:\n"
                           << "fmin = " << fmin << "\n"
                           << "fedm = " << fedm << "\n"
                           << "istat = " << istat << "\n"
                           << "sierflg = " << sierflg << "\n";    
    }

    std::vector<double> SIMPLEX_value(nPar, 0.0);
    std::vector<double> SIMPLEX_error(nPar, 0.0);
    if (try_again && smierflg != 0)
    {
        for (size_t i = 0; i < nPar; ++i)
        {
            double v, e;
            MuonResidualsGPRFitter_TMinuit->GetParameter(i, v, e);
            SIMPLEX_value[i] = v;
            SIMPLEX_error[i] = e;
        }
        Tracer::instance() << "SIMPLEX_value: ";
        Tracer::instance().copy(SIMPLEX_value.begin(), SIMPLEX_value.end());
        Tracer::instance() << "\n";
        Tracer::instance() << "SIMPLEX_error: ";
        Tracer::instance().copy(SIMPLEX_error.begin(), SIMPLEX_error.end());
        Tracer::instance() << "\n";
    }

    if (sierflg != 0)
    {
        Tracer::instance() << "Fit failed\n";
        delete MuonResidualsGPRFitter_TMinuit;
        delete fitinfo;
        return false;
    }

    if (istat != 3) 
    {
        for (int i = 0; i < 10; i++)
        {
            arglist[i] = 0.0;
        }
        ierflg = 0;
        MuonResidualsGPRFitter_TMinuit->mnexcm("HESSE", arglist, 0, ierflg);
        // MuonResidualsGPRFitter_TMinuit->mnexcm("MINOS", arglist, 0, ierflg);
    }

    MuonResidualsGPRFitter_TMinuit->mnstat(fmin, fedm, errdef, npari, nparx, istat);

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

    // std::cout << "Calculate Gradient\n";
    // std::unique_ptr<Double_t[]> min(new Double_t[6]);
    // std::unique_ptr<Double_t[]> grad(new Double_t[6]);
    // for (int i = 0; i < 6; ++i)
    // {
    //     min[i] = m_value[i];
    //     grad[i] = 0.0;
    // }

    // Int_t nPar = npar();
    // Int_t flag = 2; // 1 - calculate fval, 2 - calculate gradient; NOT WORKING!
    // Double_t fval = 0.0;

    // MuonResidualsGPRFitter_TMinuit->Eval(nPar, grad.get(), fval, min.get(), flag);

    // TMatrixDSym covMat(6);
    // MuonResidualsGPRFitter_TMinuit->mnemat(covMat.GetMatrixArray(), 6);
    // TMatrixDSym invCovMat = covMat.Invert();
    // for (int i = 0; i < 6; ++i)
    // {
    //     for (int j = 0; j < 6; ++j)
    //     {
    //         std::cout << "invCovMat["<< i << "][" << j << "] = " << invCovMat[i][j] << "\n";
    //     }
    // }
    // std::cout << "det = " << invCovMat.Determinant() << "\n";


    // std::cout << "Status codes after HESSE:\n";
    // std::cout << "istat = " << istat << "\n";
    // std::cout << "smierflg = " << smierflg << "\n";
    // std::cout << "ierflg = " << ierflg << "\n";
    // std::cout << "--------------------------\n";

    Tracer::instance() << "m_value: ";
    Tracer::instance().copy(m_value.begin(), m_value.end());
    Tracer::instance() << "\n";
    Tracer::instance() << "m_error: ";
    Tracer::instance().copy(m_error.begin(), m_error.end());
    Tracer::instance() << "\n";
    Tracer::instance() << "fmin = " << fmin << "\n"
                       << "fedm = " << fedm << "\n"
                       << "errdef = " << errdef << "\n"
                       << "npari = " << npari << "\n"
                       << "nparx = " << nparx << "\n"
                       << "istat = " << istat << "\n"
                       << "ierflg = " << ierflg << "\n"
                       << "smierflg = " << smierflg << "\n"
                       << "npar() = " << npar() << "\n";

    avg_call_duration /= FCN_calls + 1;
    Tracer::instance() << "average call duration = " << avg_call_duration << " ms\n";
    Tracer::instance() << "# FCN calls = " << FCN_calls + 1 << "\n";
    FCN_calls = 0;
    avg_call_duration = 0.0f;

    delete MuonResidualsGPRFitter_TMinuit;
    delete fitinfo;
    if (smierflg != 0 || sierflg != 0)
    {
        return false;
    }    
    return true;
}

bool MuonResidualsGPRFitter::fit()
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

    std::vector<double> steps{ 0.01,
                               0.01,
                               0.01,
                               0.0001,
                               0.0001,
                               0.0001 };

    // parameter ranges 
    std::vector<double> lows{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    std::vector<double> highs{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    return dofit(&MuonResidualsGPRFitter_FCN, nums, names, starts, steps, lows, highs);
}

void MuonResidualsGPRFitter::scanFCN(int grid_size, std::vector<double> const& lows, std::vector<double> const& highs)
{
    std::function<double(std::vector<double> const&)> fcn = [this](std::vector<double> const& parameters)
    {
        DTGeometry const* geom = this->getDTGeometry();

        int ndim = parameters.size();
        double* par = new double[ndim];
        for (int i = 0; i < ndim; ++i)
        {
            par[i] = parameters[i];
        }

        double fval = 0.0;
        for (std::map<Alignable*, MuonResidualsTwoBin*>::const_iterator chamber_data = this->datamap_begin();
                                                                        chamber_data != this->datamap_end();
                                                                        ++chamber_data)
        {
            DetId id = chamber_data->first->geomDetId();

            Alignable const* ali = chamber_data->first;
            align::PositionType const& position = ali->globalPosition();
            align::RotationType const& orientation = ali->globalRotation();

            DTChamberId cid(id.rawId());
            int station = cid.station();
            int wheel = cid.wheel();
            int sector = cid.sector();

            align::EulerAngles globAngles(3);
            globAngles[0] = par[static_cast<int>(PARAMS::kAlignPhiX)];
            globAngles[1] = par[static_cast<int>(PARAMS::kAlignPhiY)];
            globAngles[2] = par[static_cast<int>(PARAMS::kAlignPhiZ)];
            align::RotationType mtrx = align::toMatrix(globAngles);
            align::EulerAngles locAngles = align::toAngles(orientation*mtrx*orientation.transposed());

            double dx, dy, dz;
            dx = mtrx.xx()*position.x() + mtrx.yx()*position.y() + mtrx.zx()*position.z() - position.x();
            dy = mtrx.xy()*position.x() + mtrx.yy()*position.y() + mtrx.zy()*position.z() - position.y();
            dz = mtrx.xz()*position.x() + mtrx.yz()*position.y() + mtrx.zz()*position.z() - position.z();

            // double alignx_g = par[static_cast<int>(PARAMS::kAlignX)];
            // double aligny_g = par[static_cast<int>(PARAMS::kAlignY)];
            // double alignz_g = par[static_cast<int>(PARAMS::kAlignZ)];
            double alignx_g = par[static_cast<int>(PARAMS::kAlignX)] + dx;
            double aligny_g = par[static_cast<int>(PARAMS::kAlignY)] + dy;
            double alignz_g = par[static_cast<int>(PARAMS::kAlignZ)] + dz;
            // double alignphix_g = par[static_cast<int>(PARAMS::kAlignPhiX)];
            // double alignphiy_g = par[static_cast<int>(PARAMS::kAlignPhiY)];
            // double alignphiz_g = par[static_cast<int>(PARAMS::kAlignPhiZ)];

            std::vector<double> resWidths = this->getResWidths(id);
            double resXsigma = resWidths[static_cast<int>(ResidSigTypes::kResXSigma)];
            double resYsigma = resWidths[static_cast<int>(ResidSigTypes::kResYSigma)];
            double slopeXsigma = resWidths[static_cast<int>(ResidSigTypes::kResXslopeSigma)];
            double slopeYsigma = resWidths[static_cast<int>(ResidSigTypes::kResYslopeSigma)];

            GlobalVector translation_g = GlobalVector(alignx_g, aligny_g, alignz_g);
            // GlobalVector rotation_g = GlobalVector(alignphix_g, alignphiy_g, alignphiz_g);
            LocalVector translation_l = geom->idToDet(id)->toLocal(translation_g);
            // LocalVector rotation_l = geom->idToDet(id)->toLocal(rotation_g);

            double const alignx_l = translation_l.x(); 
            double const aligny_l = translation_l.y(); 
            double const alignz_l = translation_l.z();
            // double const alignphix_l = rotation_l.x();
            // double const alignphiy_l = rotation_l.y();
            // double const alignphiz_l = rotation_l.z(); 
            double const alignphix_l = locAngles[0];
            double const alignphiy_l = locAngles[1];
            double const alignphiz_l = locAngles[2]; 

            // double alignx_g = par[static_cast<int>(PARAMS::kAlignX)];
            // double aligny_g = par[static_cast<int>(PARAMS::kAlignY)];
            // double alignz_g = par[static_cast<int>(PARAMS::kAlignZ)];
            // double alignphix_g = par[static_cast<int>(PARAMS::kAlignPhiX)];
            // double alignphiy_g = par[static_cast<int>(PARAMS::kAlignPhiY)];
            // double alignphiz_g = par[static_cast<int>(PARAMS::kAlignPhiZ)];

            // std::vector<double> resWidths = this->getResWidths(id);
            // double resXsigma = resWidths[static_cast<int>(ResidSigTypes::kResXSigma)];
            // double resYsigma = resWidths[static_cast<int>(ResidSigTypes::kResYSigma)];
            // double slopeXsigma = resWidths[static_cast<int>(ResidSigTypes::kResXslopeSigma)];
            // double slopeYsigma = resWidths[static_cast<int>(ResidSigTypes::kResYslopeSigma)];

            // GlobalVector translation_g = GlobalVector(alignx_g, aligny_g, alignz_g);
            // GlobalVector rotation_g = GlobalVector(alignphix_g, alignphiy_g, alignphiz_g);
            // LocalVector translation_l = geom->idToDet(id)->toLocal(translation_g);
            // LocalVector rotation_l = geom->idToDet(id)->toLocal(rotation_g);

            // double const alignx_l = translation_l.x(); 
            // double const aligny_l = translation_l.y(); 
            // double const alignz_l = translation_l.z();
            // double const alignphix_l = rotation_l.x();
            // double const alignphiy_l = rotation_l.y();
            // double const alignphiz_l = rotation_l.z();

            if (station != 4)
            {
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
                }
            }
            else 
            {
                continue;
            }
        }
        delete [] par;
        return fval;
    };

    // std::random_device rd;
    // std::uniform_real_distribution<double> dist(-0.001, 0.001);
    // auto randNum = [&rd, &dist]() { return dist(rd); };

    // std::vector<double> lows{ -0.2, -0.2, -0.2, -0.001, -0.001, -0.001 };
    // std::vector<double> highs{ 0.2, 0.2, 0.2, 0.001, 0.001, 0.001 };
    std::vector<std::string> par_names = {"dx", "dy", "dz", "dphix", "dphiy", "dphiz"};
    std::vector<double> steps;

    int ndim = lows.size();
    // int grid_size = 100;

    if (grid_size % 2 != 0)
    {
        ++grid_size;
    }

    for (int i = 0; i < ndim; ++i)
    {
        steps.push_back((highs[i] - lows[i])/grid_size);
    }

    TFile* file = new TFile("fcn_plots.root", "RECREATE");
    TStyle* gStyle = new TStyle();

    // 2d plotting
    std::string name = "graph2d_";
    for (int par_to_plot1 = 0; par_to_plot1 < ndim; ++par_to_plot1)
    {
        for (int par_to_plot2 = par_to_plot1 + 1; par_to_plot2 < ndim; ++par_to_plot2)
        {
            name += par_names[par_to_plot1] + "_vs_" +par_names[par_to_plot2];
            std::cout << "plotting " << name << std::endl;
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
                    double fval = fcn(params);
                    xvals.push_back(var1);
                    yvals.push_back(var2);
                    fvals.push_back(fval);
                }
            }

            TGraph2D* graph2d = new TGraph2D(xvals.size(), &xvals[0], &yvals[0], &fvals[0]);
            gStyle->SetPalette(1);
            graph2d->Draw("surf1");
            file->WriteObject(graph2d, name.c_str());

            delete graph2d;
            name = "graph2d_";
        }
    }

    // 1d plotting
    name = "graph1d_";
    for (int par_to_plot = 0; par_to_plot < ndim; ++par_to_plot)
    {   
        name += par_names[par_to_plot];
        std::cout << "plotting " << name << std::endl;
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
            double fval = fcn(params);
            x.push_back(var);
            y.push_back(fval);
        }

        TGraph* graph1d = new TGraph(x.size(), &x[0], &y[0]);
        graph1d->SetLineWidth(2);
        graph1d->Draw("AL");
        file->WriteObject(graph1d, name.c_str());

        delete graph1d;
        name = "graph1d_";
    }

    file->Close();
}
