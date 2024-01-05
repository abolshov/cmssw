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
#include <unordered_map>
#include <fstream>
#include <limits>
#include <random>

struct ResidStats
{
    std::vector<double> means;
    std::vector<double> stddevs;
};

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
        kNData
    };

    enum class PARAMS {
        kAlignX,
        kAlignY,
        kAlignZ,
        kAlignPhiX,
        kAlignPhiY,
        kAlignPhiZ,
        kCount // needed to count number of minuit parameters
    };

    MuonResidualsGPRFitter();
    MuonResidualsGPRFitter(DTGeometry const* DTGeom, CSCGeometry const* CSCGeom, std::vector<std::string> const& opt);
    ~MuonResidualsGPRFitter();

    MuonResidualsGPRFitter(MuonResidualsGPRFitter const& other) = delete;
    MuonResidualsGPRFitter& operator=(MuonResidualsGPRFitter const& other) = delete;
    MuonResidualsGPRFitter(MuonResidualsGPRFitter&& other) = delete;
    MuonResidualsGPRFitter& operator=(MuonResidualsGPRFitter&& other) = delete;

    inline void SetPrintLevel(int printLevel) { m_printLevel = printLevel; }
    inline void SetStrategy(int strategy) { m_strategy = strategy; }
    void SetOptions(std::vector<std::string> const& options); 
    void SetOption(size_t optId, std::string const& option);
    inline double loglikelihood() const { return m_loglikelihood; }
    inline int npar() const { return static_cast<int>(PARAMS::kCount); }

    // methods to access residuals
    inline std::unordered_map<DetId, std::vector<double*>>::const_iterator DataBegin() const { return m_data.cbegin(); }
    inline std::unordered_map<DetId, std::vector<double*>>::const_iterator DataEnd() const { return m_data.cend(); }
    inline std::vector<double*>::const_iterator ResidualsBegin(DetId const& id) const { return m_data.at(id).cbegin(); }
    inline std::vector<double*>::const_iterator ResidualsEnd(DetId const& id) const { return m_data.at(id).cend(); }
    inline bool IsEmpty() const { return m_data.empty(); }
    void CopyData(std::map<Alignable*, MuonResidualsTwoBin*> const& from, double fraction = 1.0);

    // checks if chamber is seleced for alignment
    bool Select(DetId const& id) const;

    void ReleaseData();
    void CalcStats();
    int TrackCount() const;
    inline size_t NumberOfChambers() const { return m_data.size(); }
    size_t NTypesOfResid(DetId const& id) const;
    size_t NTypesOfResid(Alignable const* ali) const;

    inline double GetParamValue(int index) const { return m_value.at(index); }
    inline double GetParamError(int index) const { return m_error.at(index); }
    // inline std::vector<double> GetStdDev(DetId id) const { return m_stddevs.at(id); } 
    inline std::vector<double> GetStdDevs(DetId id) const { return m_stats.at(id).stddevs; } 
    inline std::vector<double> GetMeans(DetId id) const { return m_stats.at(id).means; } 

    // geometry get/set
    inline DTGeometry const* GetDTGeometry() const { return m_DTGeometry; }
    inline void SetDTGeometry(DTGeometry const* dtGeometry) { m_DTGeometry = dtGeometry; }
    inline CSCGeometry const* GetCSCGeometry() const { return m_CSCGeometry; }
    inline void SetCSCGeometry(CSCGeometry const* cscGeometry) { m_CSCGeometry = cscGeometry; }

    // debugging tools
    void Print(int nValues, DetId const& id) const;
    void Print(int nValues) const;
    void SaveResidDistr() const;
    void SaveResidPeakDistr(std::vector<double> const& params) const;
    void PlotFCN(int grid_size = 50, 
                 std::vector<double> const& lows = { -0.2, -0.2, -0.2, -0.02, -0.02, -0.02 }, 
                 std::vector<double> const& highs = { 0.2, 0.2, 0.2, 0.02, 0.02, 0.02 });
    void PlotContours(std::string selection, int n_points = 16); 
    void PlotContour(PARAMS par1, PARAMS par2, int n_points = 16); // add try-catch to Contour call

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

    // Data is stored in pairs [DetId, arrays of residuals]
    std::unordered_map<DetId, std::vector<double *>> m_data;
    // std::unordered_map<DetId, std::vector<double>> m_stddevs;
    std::unordered_map<DetId, ResidStats> m_stats;

    // encode which chambers to select for alignment
    enum Options { DTWheels, DTStations, CSCEndcaps, CSCRings, CSCStations, OptCount };
    std::vector<std::string> m_options;

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
    MuonResidualsGPRFitterFitInfo(MuonResidualsGPRFitter* gpr_fitter) : m_gpr_fitter(gpr_fitter) {}
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

#endif //MUON_RESIDUALS_GPR_FITTER_H