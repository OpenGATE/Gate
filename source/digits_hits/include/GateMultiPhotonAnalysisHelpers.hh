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
 *  About namespace: Helper functions for multi-photon analysis, including structures holding
 *  gamma interaction statistics and functions accumulating them from interaction process names.
 **/
namespace MultiPhotonAnalysisHelpers {

    /** @brief Default name stored when no scattering volume has been recorded yet. */
    inline const std::string kNoVolumeName = "NULL";

    /**
     * @brief Phantom scattering counters of a single gamma, summed over the whole event.
     *
     * GateAnalysis counts phantom interactions in a loop that runs before crystal hits are
     * processed, so every crystal hit of a given gamma receives the same event total. These
     * counters reproduce that behaviour.
     */
    struct PhantomStatistics {
        int compton = 0;
        int rayleigh = 0;
        std::string comptonVolumeName = kNoVolumeName;
        std::string rayleighVolumeName = kNoVolumeName;
    };

    /**
     * @brief Running counters of a single gamma, accumulated along the event time axis.
     *
     * Crystal counters reproduce GateAnalysis: they grow hit by hit and include the current
     * hit. `scatters` feeds nInteractions and counts Compton and Rayleigh interactions in both
     * the phantom and the crystal.
     */
    struct RunningStatistics {
        int crystalCompton = 0;
        int crystalRayleigh = 0;
        int scatters = 0;
    };

    /** @brief Interaction process classes used by multi-photon analysis. */
    enum class InteractionProcess {
        Other,
        Compton,
        Rayleigh
    };

    /**
     * @brief Classifies process name into a compact process category.
     *
     * Matching is done by substring, exactly like GateAnalysis (`find("ompt")` and
     * `find("Rayl")`), so names such as `compt`, `Compton` or `LowEnCompt` are all recognized.
     *
     * Args:
     *   processName: Geant4 process name.
     *
     * Returns:
     *   Recognized interaction category or Other.
     */
    constexpr InteractionProcess GetInteractionProcess(const std::string_view processName) {
        if (processName.find("ompt") != std::string_view::npos) {
            return InteractionProcess::Compton;
        }
        if (processName.find("Rayl") != std::string_view::npos) {
            return InteractionProcess::Rayleigh;
        }
        return InteractionProcess::Other;
    }

    /**
     * @brief Adds a phantom interaction to the event totals of one gamma.
     *
     * Args:
     *   processName: Geant4 process name at the phantom hit.
     *   volumeName: Name of the volume the phantom hit occurred in.
     *   phantomStatistics: Statistics object to update.
     */
    inline void AccumulatePhantomTotals(const std::string_view processName,
                                        const std::string& volumeName,
                                        PhantomStatistics& phantomStatistics) {
        switch (GetInteractionProcess(processName)) {
            case InteractionProcess::Compton:
                phantomStatistics.compton++;
                phantomStatistics.comptonVolumeName = volumeName;
                break;
            case InteractionProcess::Rayleigh:
                phantomStatistics.rayleigh++;
                phantomStatistics.rayleighVolumeName = volumeName;
                break;
            default:
                break;
        }
    }

    /**
     * @brief Accumulates a crystal interaction into the running counters.
     *
     * Args:
     *   processName: Geant4 process name at the crystal hit.
     *   runningStatistics: Statistics object to update.
     */
    inline void AccumulateCrystal(const std::string_view processName,
                                  RunningStatistics& runningStatistics) {
        switch (GetInteractionProcess(processName)) {
            case InteractionProcess::Compton:
                runningStatistics.crystalCompton++;
                runningStatistics.scatters++;
                break;
            case InteractionProcess::Rayleigh:
                runningStatistics.crystalRayleigh++;
                runningStatistics.scatters++;
                break;
            default:
                break;
        }
    }

    /**
     * @brief Accumulates a phantom interaction into the running counters.
     *
     * Only the nInteractions counter is affected - phantom Compton and Rayleigh columns are
     * event totals and are computed separately (see PhantomStatistics).
     *
     * Args:
     *   processName: Geant4 process name at the phantom hit.
     *   runningStatistics: Statistics object to update.
     */
    inline void AccumulatePhantom(const std::string_view processName,
                                  RunningStatistics& runningStatistics) {
        switch (GetInteractionProcess(processName)) {
            case InteractionProcess::Compton:
            case InteractionProcess::Rayleigh:
                runningStatistics.scatters++;
                break;
            default:
                break;
        }
    }

}// namespace MultiPhotonAnalysisHelpers


#endif // GateMultiPhotonAnalysisHelpers_h
