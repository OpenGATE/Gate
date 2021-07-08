/*-------------------------------------------------------

List Mode Format 
                        
--  exempleMain_04.c  --                      

Luc.Simon@iphe.unil.ch
Martin.Rey@epfl.ch

Crystal Clear Collaboration
Copyright (C) 2003 IPHE/UNIL, CH-1015 Lausanne

This software is distributed under the terms 
of the GNU Lesser General 
Public Licence (LGPL)
See LMF/LICENSE.txt for further details

-------------------------------------------------------*/
/*------------------------------------------------------------------------

			   
Description : 
Just a reader, but more user friendly


---------------------------------------------------------------------------*/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "../includes/lmf.h"
#include <time.h>




clock_t startClock, endClock;
time_t startTime, endTime;
FILE *pftime;

int main(int argc, char *argv[])
{
  char *inputName;
  /*    i8 nameOfCCSFile[81];  */
  int choice = 1, choice2 = 1;
  u16 nb, nbOfSct = 0;
  int lowerLimit = 250, upperLimit = 750;
  u16 *Rsector = NULL;
  i64 *shiftedTime = NULL;
  char response[10];
  u32 clkRatio = 0X40000;
  u32 angularShift = 30;
  u32 axialShift = 60;

  u16 nbOfSectors = 0, nbOfModules = 0, nbOfCrystals = 0;
  u16 *sector = NULL, *module = NULL, *crystal = NULL;
  u16 sctNb = 2;

  /*    pftime = fopen("time.txt","w"); */
  printf("\n****************************************\n");
  printf("                EXEMPLE 4                ");
  printf("\n****************************************\n\n\n");

  if (argc == 2) {
    inputName = argv[1];
    if (LMFcchReader(inputName))
      exit(EXIT_FAILURE);
  }
  else {
    system("ls *.cch *.ccs");
    if (LMFcchReader(""))
      exit(EXIT_FAILURE);		/* read file.cch and fill in structures LMF_cch */
  }

  while (choice != 0) {
    printf("<ENTER> to continue \n");
    getchar();
    printf("\n\n\nMenu : \n");
    printf("1. Count records in a .ccs file\n");
    printf("2. Dump a .ccs file \n");
    printf("3. Sort coincidences in a .ccs file\n");
    printf("4. Copy and treat .ccs file\n");
    printf("5. Extract xyz coord. of event records of a .ccs file\n");
    printf("6. Apply an energy window \n");
    printf("7. Sort chronologically a singles file\n");
    printf("8. Analyse coincidences (for coinci. file only)\n");
    printf("9. Output in Ascii file \n");
    printf("10. Output in a ROOT file \n");
    printf
	("11. Delay time for event comming from a choosen Rsector in a .ccs file\n");
    printf
	("12. Associate Gantry positions coming from a binary file writeCnt.bin of a .ccs file\n");
    printf("13. Mix the event coming from different block\n");
    printf("14. Keep events only from specified geometric position\n");
    printf("15. Correct DAQ time\n");
    printf("16. Sort multiple coincidences in a .ccs file\n");
    printf("17. Set a fixed axial shift\n");
    printf("18. Add angular shift\n");
    printf("19. Modelized the DAQ buffer comportement\n");
    printf("20. Follow count rates by gaussian distrib or min\n");
    printf("0. Exit\n\n\n");

    choice = hardgeti16(0, 20);

    if (choice == 1) {

      LMFreader(ccsFileName, "countRecords");
      /*             LMFreader("./ccs_files/testlmfCoinci2.ccs","countRecords"); */
      LMFreaderDestructor();
    }
    if (choice == 2) {

      LMFreader(ccsFileName, "dump");
      LMFreaderDestructor();
    }
    if (choice == 3) {


      startClock = clock();
      startTime = time(NULL);


      get_extension_ccs_FileName("_coinci");

      LMFreader(ccsFileName, "sortCoincidence");
      LMFreaderDestructor();


      endClock = clock();
      endTime = time(NULL);
      /*     fprintf(pftime,"\nTime = %ld\t",endTime - startTime); */
      /*     fprintf(pftime,"Clock = %ld\n",(endClock - startClock)/1000); */

    }
    if (choice == 4) {

      get_extension_ccs_FileName("_bis");



      printf("Treatment on singles available :\n");
      printf("1. Dead time original model\n");
      printf("2. Juelich dead time model\n");
      printf("3. Delay line\n");

      choice = hardgeti16(1, 3);

      if (choice == 1) {

	setDeadTimeMode();
	/* mode, dead time (picoseconds), ratio */
	/* mode 0 no dead time  */
	/* mode 1 paralysable */
	/* mode 2 non paralysable */
	/* mode 3 combinaison with a part  */
	/* of paralysable = ratio (0-1) */
	LMFreader(ccsFileName, "treatAndCopy");
      }

      if (choice == 2) {
	LMFreader(ccsFileName, "juelichDeadTime");
      }
      if (choice == 3) {

	LMFreader(ccsFileName, "delayLine");
      }


      LMFreaderDestructor();

    }


    if (choice == 5) {

      LMFreader(ccsFileName, "locateIdInScanner");
      LMFreaderDestructor();
    }

    if (choice == 6) {

      get_extension_ccs_FileName("_bis");

      printf("Lower energy limit (%d keV): ", lowerLimit);
      if (fgets(response, sizeof(response), stdin))
	sscanf(response, "%d", &lowerLimit);
      setDownEnergyLimit(lowerLimit);

      printf("Upper energy limit (%d keV): ", upperLimit);
      if (fgets(response, sizeof(response), stdin))
	sscanf(response, "%d", &upperLimit);
      setUpEnergyLimit(upperLimit);

      //setFPGANeighSelect(99);

      LMFreader(ccsFileName, "energyWindow");
      LMFreaderDestructor();

    }
    if (choice == 7) {
      get_extension_ccs_FileName("_bis");
      setTimeListSize(1000);
      LMFreader(ccsFileName, "sortTime");


      /* LMFreaderDestructor();  */
    }

    if (choice == 8) {
      LMFreader(ccsFileName, "analyseCoinci");
      LMFreaderDestructor();
    }
    if (choice == 9) {
      LMFreader(ccsFileName, "outputAscii");
      LMFreaderDestructor();
    }

    if (choice == 10) {
      LMFreader(ccsFileName, "outputRoot");
      LMFreaderDestructor();
    }

    if (choice == 11) {
      get_extension_ccs_FileName("_bis");

      printf("How many sectors do you want to delay: ");
      if (fgets(response, sizeof(response), stdin))
	sscanf(response, "%hu", &nbOfSct);
      Rsector = malloc(nbOfSct * sizeof(u16));
      shiftedTime = malloc(nbOfSct * sizeof(i64));
      for (nb = 0; nb < nbOfSct; nb++) {
	printf("Select the sector to be delayed: ");
	scanf("%hu", &(Rsector[nb]));
	getchar();
	printf("Set the delay you want to apply to sector %hu: ",
	       Rsector[nb]);
	scanf("%lld", &(shiftedTime[nb]));
	getchar();
      }
      printf("\n*****************************************\n");
      for (nb = 0; nb < nbOfSct; nb++)
	printf("  Applying %lld delay time to Rsector %hu\n",
	       shiftedTime[nb], Rsector[nb]);
      printf("*****************************************\n");

      setShiftedTime(nbOfSct, &Rsector, &shiftedTime);

      LMFreader(ccsFileName, "shiftTime");
      LMFreaderDestructor();

      if (Rsector)
	free(Rsector);
      if (shiftedTime)
	free(shiftedTime);
    }

    if (choice == 12) {
      get_extension_ccs_FileName("_bis");

      setGantryPosMode(1);
      printf
	  ("Select the clock ratio between Events clock and Gantry clock (Default %lu): ",
	   clkRatio);
      if (fgets(response, sizeof(response), stdin))
	sscanf(response, "%lu", &clkRatio);

      printf
	  ("Set the time delay (ms) between the clock read and the angular position read (Default %lu): ",
	   angularShift);
      if (fgets(response, sizeof(response), stdin))
	sscanf(response, "%lu", &angularShift);

      printf
	  ("Set the time delay (ms) between the clock read and the axial position read (Default %lu): ",
	   axialShift);
      if (fgets(response, sizeof(response), stdin))
	sscanf(response, "%lu", &axialShift);

      setClkRatio(clkRatio);
      setPositionsShifts(angularShift, axialShift);

      if (readPosFromFile("writeCnt.bin")) {
	printf("Can't read writeCnt.bin\n");
	exit(EXIT_FAILURE);
      }

      LMFreader(ccsFileName, "setGantryPosition");
      LMFreaderDestructor();

    }
    if (choice == 13) {
      get_extension_ccs_FileName("_bis");

      printf("Set the number of sectors in use (Default %hu): ", sctNb);
      if (fgets(response, sizeof(response), stdin))
	sscanf(response, "%hu", &sctNb);

      setNbOfSct(sctNb);
      LMFreader(ccsFileName, "sortBlocks");
      LMFreaderDestructor();
    }

    if (choice == 14) {
      get_extension_ccs_FileName("_bis");

      while (choice2 != 0) {
	printf("<ENTER> to continue \n");
	getchar();
	printf("\nGometric structure : \n");
	printf("0. No selection more\n");
	printf("1. Sectors\n");
	printf("2. Modules\n");
	printf("3. Crystals\n");

	choice2 = hardgeti16(0,3);

	if (choice2 == 1) {
	  printf("How many sectors do you want to keep: ");
	  if (fgets(response, sizeof(response), stdin))
	    sscanf(response, "%hu", &nbOfSectors);
	  sector = malloc(nbOfSectors * sizeof(u16));
	  for (nb = 0; nb < nbOfSectors; nb++) {
	    printf("Number of sector to keep:\n");
	    scanf("%hu", &(sector[nb]));
	  }
	  setSectors(nbOfSectors, &sector);
	}
	if (choice2 == 2) {
	  printf("How many modules do you want to keep: ");
	  if (fgets(response, sizeof(response), stdin))
	    sscanf(response, "%hu", &nbOfModules);
	  module = malloc(nbOfModules * sizeof(u16));
	  for (nb = 0; nb < nbOfModules; nb++) {
	    printf("Number of module to keep:\n");
	    scanf("%hu", &(module[nb]));
	  }
	  setModules(nbOfModules, &module);
	}
	if (choice2 == 3) {
	  printf("How many crystals do you want to keep: ");
	  if (fgets(response, sizeof(response), stdin))
	    sscanf(response, "%hu", &nbOfCrystals);
	  crystal = malloc(nbOfCrystals * sizeof(u16));
	  for (nb = 0; nb < nbOfCrystals; nb++) {
	    printf("Number of crystal to keep:\n");
	    scanf("%hu", &(crystal[nb]));
	  }
	  setCrystals(nbOfCrystals, &crystal);
	}
      }
      LMFreader(ccsFileName, "geometrySelector");

      LMFreaderDestructor();
      if (sector)
	free(sector);
      if (module)
	free(module);
      if (crystal)
	free(crystal);
    }

    if (choice == 15) {
      get_extension_ccs_FileName("_bis");

      printf("Set the number of sectors in use (Default %hu): ", sctNb);
      if (fgets(response, sizeof(response), stdin))
	sscanf(response, "%hu", &sctNb);

      setNbOfDaqSct(sctNb);

      LMFreader(ccsFileName, "correctDaqTime");
      LMFreaderDestructor();
    }

    if (choice == 16) {
      get_extension_ccs_FileName("_coinci");

      setMultipleCoincidencesSorterParams();
      LMFreader(ccsFileName, "sortMultiCoincidences");
      LMFreaderDestructor();
    }

    if (choice == 17) {
      get_extension_ccs_FileName("_bis");
      setNewAxialPos();

      LMFreader(ccsFileName, "changeAxialPos");
      LMFreaderDestructor();
    }

    if (choice == 18) {
      get_extension_ccs_FileName("_bis");
      setNewAngularPos();

      LMFreader(ccsFileName, "changeAngularPos");
      LMFreaderDestructor();
    }

    if (choice == 19) {
      get_extension_ccs_FileName("_bis");
      LMFreader(ccsFileName, "daqBuffer");
      LMFreaderDestructor();
    }

    if (choice == 20) {
      get_extension_ccs_FileName("_bis");
      setInitialParamsForFollowCountRates();

      LMFreader(ccsFileName, "followCountRates");
      LMFreaderDestructor();
    }
  }


  LMFcchReaderDestructor();
  /*-=-=-=--=-=-=--=-=-=-==----=-==-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/


  /*       fclose(pftime);   */
  printf("Main over\n");





  return (EXIT_SUCCESS);


}
