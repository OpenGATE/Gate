/** ----------------------
  Copyright (C): OpenGATE Collaboration
  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GateMultiPhotonAnalysisHelpers_h
#define GateMultiPhotonAnalysisHelpers_h

#include <string>
#include <string_view>

/** Authors: Wojciech Krzemień, Mateusz Bała and Kamil Dulski
 *  Emails: wojciech.krzemien@ncbj.gov.pl, mateusz.bala@ncbj.gov.pl and kamil.dulski@gmail.com
 *  Organization: National Centre For Nuclear Research (NCBJ, https://ncbj.gov.pl), Poland
 *  Developed within the IMPET project: https://pet.ncbj.gov.pl/
 *  About namespace: Helper functions for multi-photon analysis, including a structure to hold gamma interaction statistics and functions to accumulate statistics based on interaction process names.
 **/
namespace MultiPhotonAnalysisHelpers {

    /** @brief Aggregated interaction counters for a single tracked gamma. */
    struct GammaStatistics {
        int phantomCompton = 0;
        int phantomRayleigh = 0;
        int crystalCompton = 0;
        int crystalRayleigh = 0;
        int phantomInteractions = 0;
        int crystalInteractions = 0;
        std::string comptonVolumeName = "NULL";
        std::string rayleighVolumeName = "NULL";
    };

    /** @brief Interaction process classes used by multi-photon analysis. */
    enum class InteractionProcess {
        Unknown,
        Compton,
        Rayleigh,
        Transportation
    };

    /**
     * @brief Classifies process name into a compact process category.
     *
     * Args:
     *   processName: Geant4 process name.
     *
     * Returns:
     *   Recognized interaction category or Unknown.
     */
    constexpr InteractionProcess GetInteractionProcess(const std::string_view processName) {
        if (processName.size() >= 4) {
            if (processName.substr(0, 4) == "Tran") {
                return InteractionProcess::Transportation;
            }
            if (processName.substr(0, 4) == "Comp" || processName.substr(0, 4) == "comp") {
                return InteractionProcess::Compton;
            }
            if (processName.substr(0, 4) == "Rayl" || processName.substr(0, 4) == "rayl") {
                return InteractionProcess::Rayleigh;
            }
        }
        return InteractionProcess::Unknown;
    }

    /**
     * @brief Accumulates crystal-side interaction statistics.
     *
     * Args:
     *   processName: Geant4 process name at crystal hit.
     *   gammaStatistics: Statistics object to update.
     */
    inline void AccumulateCrystal(const std::string_view processName, GammaStatistics& gammaStatistics) {
        const InteractionProcess process = GetInteractionProcess(processName);
        if (process != InteractionProcess::Transportation) {
            gammaStatistics.crystalInteractions++;
            switch (process) {
                case InteractionProcess::Compton:
                    gammaStatistics.crystalCompton++;
                    break;
                case InteractionProcess::Rayleigh:
                    gammaStatistics.crystalRayleigh++;
                    break;
                default:
                    break;
            }
        }
    }

    /**
     * @brief Accumulates phantom-side interaction statistics.
     *
     * Args:
     *   processName: Geant4 process name at phantom hit.
     *   gammaStatistics: Statistics object to update.
     */
    inline void AccumulatePhantom(const std::string_view processName, GammaStatistics& gammaStatistics) {
        const InteractionProcess process = GetInteractionProcess(processName);
        if (process != InteractionProcess::Transportation) {
            gammaStatistics.phantomInteractions++;
            switch (process) {
                case InteractionProcess::Compton:
                    gammaStatistics.phantomCompton++;
                    break;
                case InteractionProcess::Rayleigh:
                    gammaStatistics.phantomRayleigh++;
                    break;
                default:
                    break;
            }
        }
    }

}// namespace MultiPhotonAnalysisHelpers


#endif // GateMultiPhotonAnalysisHelpers_h
