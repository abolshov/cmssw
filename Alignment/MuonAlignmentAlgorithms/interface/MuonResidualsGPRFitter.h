// #ifndef Alignment_MuonAlignmentAlgorithms_MuonResidualsGPRFitter_H
// #define Alignment_MuonAlignmentAlgorithms_MuonResidualsGPRFitter_H
#ifndef MUON_RESIDUALS_GPR_FITTER_H
#define MUON_RESIDUALS_GPR_FITTER_H

#ifndef STANDALONE_FITTER
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#endif

#include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsTwoBin.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

#include "TMinuit.h"
#include "TH1F.h"
#include "TProfile.h"
#include "TF1.h"
#include "TMath.h"
#include "TRandom3.h"
#include "TMatrixDSym.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <fstream>
#include <limits>
#include <random>

// struct NewMinuit : public TMinuit 
// {
//     inline void Foo() const { std::cout << "Foo!\n"; }
// };

// #ifdef STANDALONE_FITTER
// #include "MuonResidualsFitter.h"
// #else
// #include "Alignment/MuonAlignmentAlgorithms/interface/MuonResidualsFitter.h"
// #endif

class MuonResidualsGPRFitter {
public:
// enum for DT station 1,2,3 data
    enum class DataDT_6DOF {
        kResidX,
        kResidY,
        kResSlopeX,
        kResSlopeY,
        kPositionX,
        kPositionY,
        kAngleX,
        kAngleY,
        kRedChi2,
        kPz,
        kPt,
        kCharge,
        kStation,
        kWheel,
        kSector,
        kChambW,
        kChambl,
        kNData
    };

    // enum for DT station 4 data
    enum class DataDT_5DOF {
        kResid,
        kResSlope,
        kPositionX,
        kPositionY,
        kAngleX,
        kAngleY,
        kRedChi2,
        kPz,
        kPt,
        kCharge,
        kStation,
        kWheel,
        kSector,
        kChambW,
        kChambl,
        kNData
    };

    enum class DataCSC_6DOF
    {
        kResid, 
        kResSlope, 
        kPositionX, 
        kPositionY, 
        kAngleX, 
        kAngleY, 
        kRedChi2, 
        kPz, 
        kPt, 
        kCharge, 
        kNData
    };

    enum class PARAMS {
        kAlignX,
        kAlignY,
        kAlignZ,
        kAlignPhiX,
        kAlignPhiY,
        kAlignPhiZ,
        // kResidXSigma,
        // kResidYSigma,
        // kResSlopeXSigma,
        // kResSlopeYSigma,
        kCount // needed to count number of minuit parameters
    };

    MuonResidualsGPRFitter(DTGeometry const* dtGeometry = nullptr, CSCGeometry const* cscGeometry = nullptr);
    ~MuonResidualsGPRFitter() = default;

    inline void SetPrintLevel(int printLevel) { m_printLevel = printLevel; }
    inline void SetStrategy(int strategy) { m_strategy = strategy; }

    // set residual distrubion widths 
    inline void SetResWidths(std::map<DetId, std::vector<double>>& sigmas) { m_resWidths = sigmas; }

    inline double loglikelihood() const { return m_loglikelihood; }

    //returns number of parameters to be fitted
    inline int npar() const { return static_cast<int>(PARAMS::kCount); }

    // methods returning all residuals
    std::map<Alignable*, MuonResidualsTwoBin*>::const_iterator datamap_begin() const { return m_datamap.begin(); }
    std::map<Alignable*, MuonResidualsTwoBin*>::const_iterator datamap_end() const { return m_datamap.end(); }

    // methods returning widths of residual distributions
    inline std::vector<double> const& getResWidths(DetId detId) const { return m_resWidths.find(detId)->second; }

    // enumeration for better accessing widths in return values of function above
    enum class ResidSigTypes { kResXSigma, kResYSigma, kResXslopeSigma, kResYslopeSigma };

    // method filling pairs (or const_iterator) to m_datamap
    void Fill(std::map<Alignable*, MuonResidualsTwoBin*>::const_iterator it);
    inline void SetData(std::map<Alignable*, MuonResidualsTwoBin*> const& datamap) { m_datamap = datamap; }

    //returns number of all residuals
    inline size_t Size() const { return m_datamap.size(); }

    //returns GPR parameter by given index
    inline double GetParamValue(int index) const { return m_value.at(index); }

    //returns param error
    inline double GetParamError(int index) const { return m_error.at(index); }

    // geometry get/set
    inline DTGeometry const* GetDTGeometry() const { return m_DTGeometry; }
    inline void SetDTGeometry(DTGeometry const* dtGeometry) { m_DTGeometry = dtGeometry; }
    inline CSCGeometry const* GetCSCGeometry() const { return m_CSCGeometry; }
    inline void SetCSCGeometry(CSCGeometry const* cscGeometry) { m_CSCGeometry = cscGeometry; }

    void scanFCN(int grid_size, std::vector<double> const& lows, std::vector<double> const& highs);

    // wrapper-function for only configuring and preparing parameters passed to dofit
    bool Fit();

private:
    // pointer to DT geometry to access methods for coordinate conversion in FCN
    DTGeometry const* m_DTGeometry;
    CSCGeometry const* m_CSCGeometry;

    int m_printLevel;
    int m_strategy;
    // bool m_weightAlignment;

    // for MINUIT's output 
    std::vector<double> m_value;
    std::vector<double> m_error;
    // TMatrixDSym m_cov;
    double m_loglikelihood;

    // map store all pairs alignable chamber - TwoBin with residuals for this chamber
    std::map<Alignable*, MuonResidualsTwoBin*> m_datamap;

    // widths of residual distributions
    std::map<DetId, std::vector<double>> m_resWidths;

    void inform(TMinuit *tMinuit); // add this later

    // function actually calculating the shift; to be called from fit function above
    // I don not understand usage of parNum and parName yet
    // takes a function pointer to function FCN
    bool dofit(void (*fcn)(int &, double *, double &, double *, int),
               std::vector<int> &parNum,
               std::vector<std::string> &parName,
               std::vector<double> &start,
               std::vector<double> &step,
               std::vector<double> &low,
               std::vector<double> &high);
};


// Auxilliary class to get information into the fit function; Idk what its doing, copied from MuonResidualsfitter.h
class MuonResidualsGPRFitterFitInfo : public TObject {
public:
    MuonResidualsGPRFitterFitInfo(MuonResidualsGPRFitter *gpr_fitter) : m_gpr_fitter(gpr_fitter) {}
    MuonResidualsGPRFitter* gpr_fitter() { return m_gpr_fitter; } // is needed to get access to gpr fitter object in likelihod calc

private:
    MuonResidualsGPRFitter* m_gpr_fitter;
#ifdef STANDALONE_FITTER
    ClassDef(MuonResidualsGPRFitterFitInfo, 1);
#endif
};

#ifdef STANDALONE_FITTER
    ClassImp(MuonResidualsGPRFitterFitInfo);
#endif

double MuonResidualsGPRFitter_logPureGaussian(double residual, double center, double sigma);

// #endif  // Alignment_MuonAlignmentAlgorithms_MuonResidualsGPRFitter_H
#endif //MUON_RESIDUALS_GPR_FITTER_H
