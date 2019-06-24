/*----------------------
   GATE version name: gate_v...

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateMacfileParser.hh"
#include <time.h>

#include <iostream> 
#include <sstream> 

// for log
#include <cmath>
// for getenv() and system()
#include <cstdlib>

using namespace std;

// These were included by I don't know why.
// #include <sys/types.h>
// #include <sys/stat.h>

GateMacfileParser::GateMacfileParser(G4String macfileName,G4int numberOfSplits,G4int numberOfAliases,G4String* aliasesPtr)
{
	macName=macfileName;
	nSplits=numberOfSplits;
	nAliases=numberOfAliases;
	timeUnit=" -1 ";
	timeStart=-1.;
	timeStop=-1.;
	timeSlice=-1.;
	addSlice=-1.;
	addSliceBool = false;
	readSliceBool = false;
	lambda=-1;
	for(int i=0;i<nAliases;i++)listOfUsedAliases.push_back(false);
	for(int i=0;i<nAliases;i++)	listOfAliases.push_back(aliasesPtr[i]);
	oldSplitNumber=-1;

	//root,ascii are enabled by default but we skeep that in cluster mode
	for(int i=0;i<SIZE;i++) enable[i]=2; // 2 is when no enable or disable commands have been found
	for(int i=0;i<SIZE;i++) filenames[i]=0;
	enable[DAQ]=0;
	enable[MDB]=0;
	PWD=getenv("PWD");

	// For the random engine's seeds
        srand(time(NULL)/**getpid()*/);
}

GateMacfileParser::~GateMacfileParser()
{
}

G4int GateMacfileParser::GenerateResolvedMacros(G4String directory)
{
	size_t pos_slash = macName.find_last_of("/");
	G4String tmp = "";
	if (pos_slash!=string::npos) tmp = macName.substr(pos_slash);
	else tmp = macName;
	size_t pos_dot = tmp.find_first_of(".");
	if (pos_dot==string::npos)
	{
		G4cerr << "***** Your macro file given in parameter does not have any extension. Please give it the .mac extension." << endl;
		exit(1);
	}
	G4String macNameDir=tmp.substr(0,pos_dot);
	G4String dir=directory+macNameDir+"/";
	ifstream dirstream(dir.c_str());
	int i=0;
	stringstream i_str; 
	while (dirstream)
	{
		i++;
		i_str.str("");
		i_str<<i;
		dirstream.close();
		dir=directory+macNameDir+i_str.str()+"/";
		dirstream.open(dir.c_str());
	}
	dirstream.close();
 
	const G4String mkdir("mkdir "+dir); 
	const int res = system(mkdir.c_str()); 
	if (res!=0) 
	{
		cout<<"failed to create directory in .Gate"<<endl;
		return 1;
	}
	if(m_verboseLevel>3) cout<<"Information : creating "<<dir<<endl;

	G4String splitfileName=dir+macNameDir+".split";
	ofstream splitfile(splitfileName.c_str());
	splitfile<<"Number of files: "<<nSplits<<endl<<endl; 

	outputDir=dir;

	for (G4int j=1;j<=nSplits;j++)
	{
		i_str.str("");
		i_str<<j;
		GenerateResolvedMacro(dir+macNameDir+i_str.str()+".mac",j,splitfile); 
		splitfile<<endl;  

		if(j%(nSplits/10)==0)
			cout<<100*j/nSplits<<"% "<<flush;
	}
	if (filenames[ROOT]==1)
		splitfile<<"Original Root filename: "<<originalRootFileName<<endl;
	splitfile.close();

	outputMacDir=dir+macNameDir;
	return 0; 
}

void GateMacfileParser::CleanAbort(ofstream& output, ofstream& splitfile)
{
	//erase created files and the directory before error exit
	if( outputDir.contains(".Gate/") )
	{ //just in case to avoid a catastrophe
		if (output) output.close();
		if (splitfile) splitfile.close();
		const G4String rmfiles="rm -f "+outputDir+"/*";
		const int res1 = system(rmfiles.c_str());
		if(res1)
		{
			G4cout << "Could not remove files " << outputDir << "/*" << endl;
			G4cout << "Please remove manually !" << endl;
		}
		const G4String rmdir="rm -f -r "+outputDir; 
		const int res2 = system(rmdir.c_str());
		if(res2)
		{
			G4cout<<"Could not remove "<<outputDir<<endl; 
			G4cout<<"Please remove manually !"<<endl;
		}
	}
}

G4String GateMacfileParser::GetOutputMacDir()
{
	return outputMacDir;
}

G4int GateMacfileParser::GenerateResolvedMacro(G4String outputName,G4int splitNumber,ofstream& splitfile)
{
	char buffer[256];
	ifstream macfile;
	const G4String dir(outputName); 
	ofstream outputMacfile(dir.c_str());
 
	macfile.open(macName);
	if (!macfile || !outputMacfile)
	{  
		cout<< "Error accessing macro input file! "<<macName<< endl; 
		return 1;
	}

	while (!macfile.eof())
	{
		macfile.getline(buffer,256);
		macline=buffer;

		if (!IsComment(macline))
		{
			FormatMacline();
			InsertAliases();
			AddAliases();
			LookForEnableOutput();
			InsertOutputFileNames(splitNumber,splitfile);
			SearchForActors(splitNumber,outputMacfile,splitfile);
			InsertSubMacros(outputMacfile,splitNumber,splitfile);
			DealWithTimeCommands(outputMacfile,splitNumber,splitfile);
			IgnoreRandomEngineCommand();
			outputMacfile<<macline<<endl; 
		}
	}
	if(splitNumber==1)
	{
		if(enable[DAQ]==0)
		{
			G4cerr<<"This macro file does not contain any startDAQ !"<<G4endl;
			CleanAbort(outputMacfile,splitfile);
			return 1;
		}
		if(enable[MDB]==0)
		{
			G4cerr<<"This macro file does not contain any MaterialDatabase !"<<G4endl;
                        CleanAbort(outputMacfile,splitfile);
			return 1;
		}
		// Output check
		G4int enabledOutput = 0;
		for (G4int i=0; i<SIZE-2; i++) if (enable[i]==1 && filenames[i]==1) enabledOutput++;
		G4cout<<"Number of enabled output: "<<enabledOutput<<endl;
		// Actor check
		G4cout << "Number of enabled actors: " << listOfEnabledActorName.size() << endl;
		// Check if no output nor actor
		if (enabledOutput+listOfEnabledActorName.size()==0) G4cerr << "***** Warning: No output module nor actor are enabled !" << endl;
		// Check if all aliases from the command line are used
		bool flag=true;
		nAliases = (G4int)listOfUsedAliases.size();
		for (G4int i=1;i<nAliases;i+=2) flag&=listOfUsedAliases[i];
		if(flag==false)
		{
			G4cerr<<"Could not use the following aliases from the command line:"<<G4endl;
			for (G4int i=1;i<nAliases;i+=2) 
				if(!listOfUsedAliases[i]) G4cout<<" "<<listOfAliases[i]<<G4endl;
			return 1; 
		}
	}
	macfile.close();
	outputMacfile.close();
	return 0;
}

void GateMacfileParser::InsertAliases()
{
	G4String insert;
	nAliases = (G4int)(listOfAliases.size());
	for (G4int i=1;i<nAliases;i+=2)
	{
		while (macline.contains("{"+listOfAliases[i]+"}"))
		{
			insert=listOfAliases[i-1];
			G4int position=macline.find("{"+listOfAliases[i]+"}",0);
			G4int length=2+listOfAliases[i].size();
			macline.replace(position,length,insert);
			listOfUsedAliases[i]=true;
		}
	}
}

void GateMacfileParser::AddAliases()
{
	if (macline.contains("/control/alias"))
	{
		G4String tmpStr = macline.substr(15,256);
		int position = tmpStr.find(" ");

		G4String aliasName(tmpStr.substr(0,position));
		for(size_t i=1;i<listOfAliases.size();i+=2)
			if(aliasName==listOfAliases[i])
				return;

		listOfAliases.push_back( tmpStr.substr(position+1,tmpStr.size()-position));
		listOfUsedAliases.push_back(false);
		nAliases++;
		listOfAliases.push_back( tmpStr.substr(0,position));
		listOfUsedAliases.push_back(false);
		nAliases++;
	}
}

void GateMacfileParser::InsertSubMacros(ofstream& output,G4int splitNumber,ofstream& splitfile)
{
	if (macline.contains("/control/execute"))
	{
		char buffer[256];
		G4String extMacfileName=macline.substr(17,256);
		if (extMacfileName.contains("/")) ExtractLocalDirectory(extMacfileName);
		ifstream extMacfile;
		extMacfile.open(extMacfileName);
		if (!extMacfile)
		{
			G4String localname=localDir+extMacfileName;
			extMacfile.clear();
			extMacfile.open(localname);
			if (!extMacfile)
			{
				cout<< "Error reading from file: "<<extMacfileName<< endl; 
				CleanAbort(output,splitfile);
				exit(1);
			}
		}
		while (!extMacfile.eof())
		{
			extMacfile.getline(buffer,256);
			macline=buffer;
			if (!IsComment(macline))
			{
				FormatMacline();
				InsertAliases();
				AddAliases();
				LookForEnableOutput();
				InsertOutputFileNames(splitNumber,splitfile);
				SearchForActors(splitNumber,output,splitfile);
				InsertSubMacros(output,splitNumber,splitfile);
				DealWithTimeCommands(output,splitNumber,splitfile);
				output<<macline<<endl;
			}
		}
		extMacfile.close();
		macline="";
	}
}

void GateMacfileParser::DealWithTimeCommands(ofstream& output,G4int splitNumber,ofstream& splitfile)
{
	if (oldSplitNumber==-1) oldSplitNumber=splitNumber;
	else if (oldSplitNumber!=splitNumber)
	{
		timeStart=-1.;
		timeStop=-1.;
		timeSlice=-1.;
		addSlice=-1.;
		lambda=-1.;
		oldSplitNumber=splitNumber;
	}
	if (macline.contains("/gate/cluster/setTimeSplitHalflife"))
	{
		if (lambda!=-1.)
		{
                        G4cerr << "***** The setTimeSplitHalfLife command is given twice !!" << endl;
                        CleanAbort(output,splitfile);
                        exit(1);
                }
		G4String subMacline=macline.substr(35,256);
		G4int position=subMacline.find(" ",0);
		G4String lambda_str=subMacline.substr(0,position);
		G4String timeUnit_tmp=subMacline.substr(position+1,subMacline.length());
		if (timeUnit==" -1 ") timeUnit=timeUnit_tmp;
		else if (timeUnit!=timeUnit_tmp)
		{
                        G4cerr << "***** Please use the same time unit for each time related to time (setTime...)" << endl;
                        CleanAbort(output,splitfile);
                        exit(1);
		}
		G4String temp=lambda_str(position-1);
		if (!strcmp(temp,".")) lambda_str=lambda_str+"0";
		stringstream lambda_ss(lambda_str);
		lambda_ss>>lambda;
		lambda=log(2.0)/lambda;
		macline="";
	}
	else if (macline.contains("/gate/application/setTimeStop"))
	{
		// We first check if we already encountered the addSlice command
		if (addSlice!=-1.)
		{
			G4cerr << "***** The setTimeStop command cannot be used with the addSlice command !" << endl;
			G4cerr << "***** When using addSlice commands, the timeStop is deduced from the total slices." << endl;
			CleanAbort(output,splitfile);
			exit(1);
		}
		if (timeStop!=-1.)
		{
			G4cerr << "***** The setTimeStop command is given twice !!" << endl;
			CleanAbort(output,splitfile);
			exit(1);
		}
		G4String subMacline=macline.substr(30,256);
		G4int position=subMacline.find(" ",0);
		G4String timeStop_str=subMacline.substr(0,position);
		G4String timeUnit_tmp=subMacline.substr(position+1,subMacline.length());
                if (timeUnit==" -1 ") timeUnit=timeUnit_tmp;
                else if (timeUnit!=timeUnit_tmp)
                {
                        G4cerr << "***** Please use the same time unit for each time related to time (setTime...)" << endl;
                        CleanAbort(output,splitfile);
                        exit(1);
                }
		G4String temp=timeStop_str(position-1);
		if (!strcmp(temp,".")) timeStop_str=timeStop_str+"0";
		stringstream timeStop_ss(timeStop_str);
		timeStop_ss>>timeStop;
		splitfile<<"Stop time is: "<<macline.substr(30,256)<<endl;
	}
	else if (macline.contains("/gate/application/setTimeStart"))
	{
		if (timeStart!=-1.)
		{
                        G4cerr << "***** The setTimeStart command is given twice !!" << endl;
                        CleanAbort(output,splitfile);
                        exit(1);
                }
		G4String subMacline=macline.substr(31,256);
		G4int position=subMacline.find(" ",0);
		G4String timeStart_str=subMacline.substr(0,position);
		G4String timeUnit_tmp=subMacline.substr(position+1,subMacline.length());
//	G4cout<< macline<<" :star= " << timeStart_str<< " :unit="<< timeUnit_tmp<< ":"<<timeUnit <<G4endl;
                if (timeUnit==" -1 ") timeUnit=timeUnit_tmp;
                else if (timeUnit!=timeUnit_tmp)
                {
                        G4cerr << "***** Please use the same time unit for each time related to time (setTime...)" << endl;
                        CleanAbort(output,splitfile);
                        exit(1);
                }
		G4String temp=timeStart_str(position-1);
		if (!strcmp(temp,".")) timeStart_str=timeStart_str+"0";
		stringstream timeStart_ss(timeStart_str);
		timeStart_ss>>timeStart;
		splitfile<<"Start time is: "<<macline.substr(31,256)<<endl;
	}
	else if (macline.contains("/gate/application/setTimeSlice"))
	{
		if (addSlice!=-1.)
		{
                        G4cerr << "***** The setTimeSlice command cannot be used in combination with the addSlice command !" << endl;
                        CleanAbort(output,splitfile);
                        exit(1);
		}
		if (timeSlice!=-1.)
		{
                        G4cerr << "***** The setTimeSlice command is given twice !!" << endl;
                        CleanAbort(output,splitfile);
                        exit(1);
                }
		G4String subMacline=macline.substr(31,256);
                G4int position=subMacline.find(" ",0);
                G4String timeSlice_str=subMacline.substr(0,position);
                G4String timeUnit_tmp=subMacline.substr(position+1,subMacline.length());
                if (timeUnit==" -1 ") timeUnit=timeUnit_tmp;
                else if (timeUnit!=timeUnit_tmp)
                {
                        G4cerr << "***** Please use the same time unit for each time related to time (setTime...)" << endl;
                        CleanAbort(output,splitfile);
                        exit(1);
                }
                G4String temp=timeSlice_str(position-1);
                if (!strcmp(temp,".")) timeSlice_str=timeSlice_str+"0";
                stringstream timeSlice_ss(timeSlice_str);
                timeSlice_ss>>timeSlice;
		splitfile<<"Timeslice is: "<<macline.substr(31,256)<<endl;  
	}
	else if (macline.contains("/gate/application/addSlice"))
        {
		if (readSliceBool)
		{
			G4cerr << "***** The commands 'addSlice' and 'readTimeSlicesIn' cannot be used together !" << endl;
			CleanAbort(output,splitfile);
                        exit(1);
		}
                if (timeSlice!=-1.)
                {
                        G4cerr << "***** The setTimeSlice command cannot be used in combination with the addSlice command !" << endl;
                        CleanAbort(output,splitfile);
                        exit(1);
                }
		if (timeStop!=-1.)
		{
                        G4cerr << "***** The setTimeStop command cannot be used with the addSlice command !" << endl;
                        G4cerr << "***** When using addSlice commands, the timeStop is deduced from the total slices." << endl;
                        CleanAbort(output,splitfile);
                        exit(1);
		}
                G4String subMacline=macline.substr(27,256);
                G4int position=subMacline.find(" ",0);
                G4String addSlice_str=subMacline.substr(0,position);
                G4String timeUnit_tmp=subMacline.substr(position+1,subMacline.length());
                if (timeUnit==" -1 ") timeUnit=timeUnit_tmp;
                else if (timeUnit!=timeUnit_tmp)
                {
                        G4cerr << "***** Please use the same time unit for each time related to time (setTime...)" << endl;
                        CleanAbort(output,splitfile);
                        exit(1);
                }
                G4String temp=addSlice_str(position-1);
                if (!strcmp(temp,".")) addSlice_str=addSlice_str+"0";
                stringstream addSlice_ss(addSlice_str);
		G4double newAddSlice;
                addSlice_ss>>newAddSlice;
		if (addSlice==-1.) addSlice = newAddSlice;
		else addSlice += newAddSlice;
                splitfile<<"New add slice: "<<macline.substr(31,256)<<endl;
		addSliceBool=true;
        }
	else if (macline.contains("/gate/application/readTimeSlicesIn"))
	{
                if (addSliceBool)
                {
                        G4cerr << "***** The commands 'addSlice' and 'readTimeSlicesIn' cannot be used together !" << endl;
                        CleanAbort(output,splitfile);
                        exit(1);
                }
                if (timeStop!=-1.)
                {
                        G4cerr << "***** The setTimeStop command cannot be used with the readTimeSlicesIn command !" << endl;
                        G4cerr << "***** When using readTimeSlicesIn commands, the timeStop is deduced from the total slices." << endl;
                        CleanAbort(output,splitfile);
                        exit(1);
                }
		G4String filename = macline.substr(35);
		// Opening file
		ifstream is;
		is.open(filename.c_str());
		if (!is)
		{
			G4cerr << "Input slice file '" << filename << "' is missing or corrupted !" << endl;
			CleanAbort(output,splitfile);
			exit(1);
		}
		skipComment(is);
		// Reading time info
		G4String timeUnit_tmp;
		if (!ReadColNameAndUnit(is, "Time", timeUnit_tmp))
		{
			G4cerr << "The file '" << filename << "' need to begin with 'Time'" << endl;
			CleanAbort(output,splitfile);
			exit(1);
		}
                if (timeUnit==" -1 ") timeUnit=timeUnit_tmp;
                else if (timeUnit!=timeUnit_tmp)
                {
                        G4cerr << "***** Please use the same time unit for each time related to time (setTime...)" << endl;
                        CleanAbort(output,splitfile);
                        exit(1);
                }
		// Beginning loop
		skipComment(is);
		double firstTime=0.;
		double lastTime=0.;
		int n=0;
		while (is)
		{
			double t;
			is >> t;
			if (n==0) firstTime = t;
			else lastTime = t;
			n++;
			skipComment(is);
		}
		is.close();
		// End reading file
		if (addSlice==-1.) addSlice = lastTime-firstTime;
                else addSlice += (lastTime-firstTime);
		splitfile<<"Global slice read in file: "<<lastTime-firstTime<<" "<<timeUnit<<endl;
		readSliceBool = true;
	}
	else if (macline.contains("gate/application/startDAQ") || macline.contains("gate/application/start"))
	{
		// to check if there is a DAQ command at all
		enable[DAQ]=1;

		// And we set a brand new seed for the random engine !
		unsigned long int theSeed = rand()*time(NULL);
		output << "/gate/random/setEngineSeed " << theSeed << endl;
 
		// after this command the job will start 
		// so all output should be defined
		if(splitNumber==1) CheckOutputPrint();
		CheckOutput(output,splitfile,splitNumber);

		// ==============================================================================================================
		// This part is here to check the presence of time commands and to deal with fu*** default
		// values permitted ... And default values always give very nice part of code, hummmmm.
		if (timeStart==-1. && timeSlice==-1. && addSlice==-1. && timeStop==-1.) // The 4 are missing
		{
			timeStart=0.;
			timeStop=1.;
			timeUnit="s";
			output << "/gate/application/setTimeStart 0. s" << endl;
			output << "/gate/application/addSlice     1. s" << endl;
		}
		else if (timeSlice==-1. && addSlice==-1. && timeStop==-1.) // Only timeStart is present
		{
			// We assume timeStop = timeStart+1
			timeStop = timeStart+1.;
			output << "/gate/application/addSlice 1. " << timeUnit << endl;
		}
		else if (timeStart==-1. && addSlice==-1. && timeStop==-1.) // Only timeSlice is present
		{
			// We assume a single run from timeStart = 0 to timeStop = timeSlice
			timeStart = 0.;
			timeStop = timeSlice;
			output << "/gate/application/setTimeStart " << timeStart << " " << timeUnit << endl;
			output << "/gate/application/setTimeStop  " << timeStop << " " << timeUnit << endl;
		}
		else if (timeStart==-1. && timeSlice==-1. && timeStop==-1.) // Only addSlice is present
		{
			// We assume timeStart = 0
			timeStart = 0.;
			timeStop = addSlice;
			output << "/gate/application/setTimeStart " << timeStart << " " << timeUnit << endl;
		}
		else if (timeStart==-1. && timeSlice==-1. && addSlice==-1.) // Only timeStop is present
		{
			// We assume a single run from timeStart = 0 to the known timeStop
			timeStart = 0.;
			output << "/gate/application/setTimeStart " << timeStart << " " << timeUnit << endl;
			output << "/gate/application/setTimeSlice " << timeStop << " " << timeUnit << endl;
		}
		else if (timeStart==-1. && addSlice==-1.) // timeStop and timeSlice are present
		{
			// We assume a single run from timeStart = timeStop-timeSlice to the known timeStop
			timeStart = timeStop - timeSlice;
			output << "/gate/application/setTimeStart " << timeStart << " " << timeUnit << endl;
		}
		else if (timeSlice==-1. && timeStop==-1.) // timeStart and addSlice are present (often the case with radiotherapist)
		{
			// We just set the timeStop to the total slice
			timeStop = addSlice;
		}
		else if (addSlice==-1. && timeStop==-1.) // timeStart and timeSlice are present
		{
			// We assume a single run from timeStart to timeStop = timeStart+timeSlice
			timeStop = timeStart + timeSlice;
			output << "/gate/application/setTimeStop  " << timeStop << " " << timeUnit << endl;
		}
		else if (addSlice==-1. && timeSlice==-1.) // timeStart and timeStop are present
		{
			// We assume a single run from timeStart to timeStop
			output << "/gate/application/setTimeSlice " << timeStop-timeStart << " " << timeUnit << endl;
		}
		else if (addSlice==-1) // timeStart, timeStop and timeSlice are present (often the case with imagist)
		{
			// Nothing to do
		}
		// All possible cases are represented here !!! Believe me !
		// ==============================================================================================================

		// Virtual time calculation
		if(timeStop-timeStart==0)
		{
			G4cout<<"***** TimeStart - TimeStop seems to be 0 ?"<<G4endl;
			CleanAbort(output,splitfile);
			exit(1);
		}
		if (lambda!=-1) CalculateTimeSplit(splitNumber);
		else
		{
			virtualStartTime=timeStart+(timeStop-timeStart)/(G4double)nSplits*(splitNumber-1);
			virtualStopTime=timeStart+(timeStop-timeStart)/(G4double)nSplits*splitNumber;
		}
		output<<"/gate/application/startDAQCluster "<<virtualStartTime<<" "<<virtualStopTime<<" "<<"0 "<<timeUnit<<endl; 
		splitfile<<"Virtual startTime: "<<virtualStartTime<<" "<<timeUnit<<endl;
		splitfile<<"Virtual stopTime: "<<virtualStopTime<<" "<<timeUnit<<endl;
		macline="";
	}
}

void GateMacfileParser::IgnoreRandomEngineCommand()
{
	if (macline.contains("/gate/random/setEngineSeed")) macline="";
}

void GateMacfileParser::CalculateTimeSplit(G4int splitNumber)
{
	G4double t1=(log((((G4double)nSplits-1.0)/(G4double)nSplits)*exp(-lambda*timeStart)+(1.0/(G4double)nSplits)*exp(-lambda*timeStop)))/(-lambda);
	if (splitNumber==1)
	{
		virtualStartTime=timeStart;
		virtualStopTime=t1;
	}
	if (splitNumber==2)
	{
		virtualStartTime=t1;
		virtualStopTime=(log((splitNumber*exp(-lambda*t1))-((splitNumber-1)*exp(-lambda*timeStart))))/(-lambda);
	}
	if (splitNumber>2)
	{
		virtualStartTime=(log(((splitNumber-1)*exp(-lambda*t1))-((splitNumber-2)*exp(-lambda*timeStart))))/(-lambda); 
		if (splitNumber!=nSplits) virtualStopTime=(log((splitNumber*exp(-lambda*t1))-((splitNumber-1)*exp(-lambda*timeStart))))/(-lambda);
		else virtualStopTime=timeStop;
	}
}

void GateMacfileParser::ExtractLocalDirectory(G4String macfileName)
{
	G4String subString;
	G4int size=macfileName.length();
	subString=macfileName.substr(size-1,256);
	G4int i=size;
	while (!subString.contains("/"))
	{
		i--;
		subString=macfileName.substr(i,256);
	}
	localDir=macfileName.substr(0,i+1);
}

/*just note if there is an enable command*/
void GateMacfileParser::LookForEnableOutput()
{
	// {ROOT=0,ASCII=1,ARF=2,PROJ=3,ECAT=4,SINO=5,ACCEL=6,LMF=7,CT=8,DAQ=9,MDB=10,SIZE=11,
	//GPUSPECT=12}
	if      (macline.contains("/gate/output/root/disable"))            {enable[ROOT]=0;macline="";}
	else if (macline.contains("/gate/output/root/enable"))             {enable[ROOT]=1;macline="";}
	else if (macline.contains("/gate/output/ascii/disable"))           {enable[ASCII]=0;macline="";}
	else if (macline.contains("/gate/output/ascii/enable"))            {enable[ASCII]=1;macline="";}
        else if (macline.contains("/gate/output/arf/disable"))             {enable[ARF]=0;macline="";}
        else if (macline.contains("/gate/output/arf/enable"))              {enable[ARF]=1;macline="";}
	else if (macline.contains("/gate/output/projection/disable"))      {enable[PROJ]=0;macline="";}
	else if (macline.contains("/gate/output/projection/enable"))       {enable[PROJ]=1;macline="";}
        else if (macline.contains("/gate/output/ecat7/disable"))           {enable[ECAT]=0;macline="";}
        else if (macline.contains("/gate/output/ecat7/enable"))            {enable[ECAT]=1;macline="";}
        else if (macline.contains("/gate/output/sinogram/disable"))        {enable[SINO]=0;macline="";}
        else if (macline.contains("/gate/output/sinogram/enable"))         {enable[SINO]=1;macline="";}
        else if (macline.contains("/gate/output/sinoAccel/disable"))       {enable[ACCEL]=0;macline="";}
        else if (macline.contains("/gate/output/sinoAccel/enable"))        {enable[ACCEL]=1;macline="";}
	else if (macline.contains("/gate/output/lmf/disable"))             {enable[LMF]=0;macline="";}
	else if (macline.contains("/gate/output/lmf/enable"))              {enable[LMF]=1;macline="";}
	else if (macline.contains("/gate/output/imageCT/disable"))         {enable[CT]=0;macline="";}
	else if (macline.contains("/gate/output/imageCT/enable"))          {enable[CT]=1;macline="";}
	else if (macline.contains("/gate/output/spectGPU/disable"))         {enable[GPUSPECT]=0;macline="";}
	else if (macline.contains("/gate/output/spectGPU/enable"))          {enable[GPUSPECT]=1;macline="";}
}

void GateMacfileParser::CheckOutputPrint()
{
	// ===========================================================================================
	// Concerning output modules
	// ===========================================================================================
	// {ROOT=0,ASCII=1,ARF=2,PROJ=3,ECAT=4,SINO=5,ACCEL=6,LMF=7,CT=8,DAQ=9,MDB=10,SIZE=11,
	//GPUSPECT=12}
	//cases are:
	//1) output disabled and not filename is given -> ok 
	//2) output disabled but a filename is given -> output is let disabled
	//3) output enabled but no filename is given -> output is disabled
	//4) output enabled and filename given -> ok
	cout << "Summary of all outputs:" << endl;
	if (enable[ROOT]==1) // the enable command was found
	{
		if (filenames[ROOT]==1) cout << "  ROOT       output is enabled" << endl; // a filaname was given, it's good
		else cout << "  ROOT       output is enabled but no filename is given (disable it by default)" << endl; // no filename given
	}
	else if (filenames[ROOT]==1) cout << "  ROOT       output is disabled but a filename is given (let it disable)" << endl;
	else cout << "  ROOT       output is disabled" << endl;
        if (enable[ASCII]==1) // the enable command was found
        {
                if (filenames[ASCII]==1) cout << "  ASCII      output is enabled" << endl; // a filaname was given, it's good
                else cout << "  ASCII      output is enabled but no filename is given (disable it by default)" << endl; // no filename given
        }
        else if (filenames[ASCII]==1) cout << "  ASCII      output is disabled but a filename is given (let it disable)" << endl;
        else cout << "  ASCII      output is disabled" << endl;
        if (enable[ARF]==1) // the enable command was found
        {
                if (filenames[ARF]==1) cout << "  ARF        output is enabled" << endl; // a filaname was given, it's good
                else cout << "  ARF        output is enabled but no filename is given (disable it by default)" << endl; // no filename given
        }
        else if (filenames[ARF]==1) cout << "  ARF        output is disabled but a filename is given (let it disable)" << endl;
        else cout << "  ARF        output is disabled" << endl;
        if (enable[PROJ]==1) // the enable command was found
        {
                if (filenames[PROJ]==1) cout << "  PROJECTION output is enabled" << endl; // a filaname was given, it's good
                else cout << "  PROJECTION output is enabled but no filename is given (disable it by default)" << endl; // no filename given
        }
        else if (filenames[PROJ]==1) cout << "  PROJECTION output is disabled but a filename is given (let it disable)" << endl;
        else cout << "  PROJECTION output is disabled" << endl;
        if (enable[ECAT]==1) // the enable command was found
        {
                if (filenames[ECAT]==1) cout << "  ECAT7      output is enabled" << endl; // a filaname was given, it's good
                else cout << "  ECAT7      output is enabled but no filename is given (disable it by default)" << endl; // no filename given
        }
        else if (filenames[ECAT]==1) cout << "  ECAT7      output is disabled but a filename is given (let it disable)" << endl;
        else cout << "  ECAT7      output is disabled" << endl;
        if (enable[SINO]==1) // the enable command was found
        {
                if (filenames[SINO]==1) cout << "  SINOGRAM   output is enabled" << endl; // a filaname was given, it's good
                else cout << "  SINOGRAM   output is enabled but no filename is given (disable it by default)" << endl; // no filename given
        }
        else if (filenames[SINO]==1) cout << "  SINOGRAM   output is disabled but a filename is given (let it disable)" << endl;
        else cout << "  SINOGRAM   output is disabled" << endl;
        if (enable[ACCEL]==1) // the enable command was found
        {
                if (filenames[ACCEL]==1) cout << "  SINOACCEL  output is enabled" << endl; // a filaname was given, it's good
                else cout << "  SINOACCEL  output is enabled but no filename is given (disable it by default)" << endl; // no filename given
        }
        else if (filenames[ACCEL]==1) cout << "  SINOACCEL  output is disabled but a filename is given (let it disable)" << endl;
        else cout << "  SINOACCEL  output is disabled" << endl;
        if (enable[LMF]==1) // the enable command was found
        {
                if (filenames[LMF]==1) cout << "  LMF        output is enabled" << endl; // a filaname was given, it's good
                else cout << "  LMF        output is enabled but no filename is given (disable it by default)" << endl; // no filename given
        }
        else if (filenames[LMF]==1) cout << "  LMF        output is disabled but a filename is given (let it disable)" << endl;
        else cout << "  LMF        output is disabled" << endl;
        if (enable[CT]==1) // the enable command was found
        {
                if (filenames[CT]==1) cout << "  CT         output is enabled" << endl; // a filaname was given, it's good
                else cout << "  CT         output is enabled but no filename is given (disable it by default)" << endl; // no filename given
        }
        else if (filenames[CT]==1) cout << "  CT         output is disabled but a filename is given (let it disable)" << endl;
        else cout << "  CT         output is disabled" << endl;
				if (enable[GPUSPECT]==1) // the enable command was found
        {
                if (filenames[GPUSPECT]==1) cout << "  GPUSPECT   output is enabled" << endl; // a filaname was given, it's good
                else cout << "  GPUSPECT   output is enabled but no filename is given (disable it by default)" << endl; // no filename given
        }
        else if (filenames[GPUSPECT]==1) cout << "  GPUSPECT   output is disabled but a filename is given (let it disable)" << endl;
        else cout << "  GPUSPECT   output is disabled" << endl;
	// ===========================================================================================
	// Concerning actors
	// ===========================================================================================
	cout << "Summary of all actors:" << endl;
	for (size_t i=0; i<listOfEnabledActorName.size(); i++) cout << "  Actor '" << listOfEnabledActorName[i] << "' of type " << listOfEnabledActorType[i] << " is enabled" << endl;
	for (size_t i=0; i<listOfActorName.size(); i++) cout << "  Actor '" << listOfActorName[i] << "' was declared but not planed to be saved" << endl;
	if (listOfActorName.size()+listOfEnabledActorName.size()==0) cout << "  ~~~" << endl;
}

// This function just add the enable command for the output validated
void GateMacfileParser::CheckOutput(ofstream& output,ofstream& /*splitfile*/,G4int /*splitNumber*/)
{
	// {ROOT=0,ASCII=1,ARF=2,PROJ=3,ECAT=4,SINO=5,ACCEL=6,LMF=7,CT=8,DAQ=9,MDB=10,SIZE=11}
	if(enable[ROOT]==1 && filenames[ROOT]==1)   output << "/gate/output/root/enable" << endl;
	if(enable[ASCII]==1 && filenames[ASCII]==1) output << "/gate/output/ascii/enable" << endl;
	if(enable[ARF]==1 && filenames[ARF]==1)     output << "/gate/output/arf/enable" << endl;
	if(enable[PROJ]==1 && filenames[PROJ]==1)   output << "/gate/output/projection/enable" << endl;
	if(enable[ECAT]==1 && filenames[ECAT]==1)   output << "/gate/output/ecat7/enable" << endl;
	if(enable[SINO]==1 && filenames[SINO]==1)   output << "/gate/output/sinogram/enable" << endl;
	if(enable[ACCEL]==1 && filenames[ACCEL]==1) output << "/gate/output/sinoAccel/enable" << endl;
	if(enable[LMF]==1 && filenames[LMF]==1)     output << "/gate/output/lmf/enable" << endl;
	if(enable[CT]==1 && filenames[CT]==1)       output << "/gate/output/imageCT/enable" << endl;
	if(enable[GPUSPECT]==1 && filenames[GPUSPECT]==1)       output << "/gate/output/spectGPU/enable" << endl;
}

//CHANGED$
void GateMacfileParser::AddPWD(G4String key)
{
	G4String filename=ExtractFileName(key);
	long unsigned int pos=filename.rfind("/");
	stringstream ss;
	if((pos!=string::npos && pos==key.length()-1) || pos==string::npos) ss<<key<<" "<<filename;
	else ss<<key<<" "<<filename;
        //if((pos!=string::npos && pos==key.length()-1) || pos==string::npos) ss<<key<<" "<<filename;
	//else ss<<key<<" "<<PWD<<"/"<<filename;
	macline=ss.str();
}

/*assumes that aliasing has been completed beforehand*/
void GateMacfileParser::InsertOutputFileNames(G4int splitNumber,ofstream& splitfile)
{
	// {ROOT=0,ASCII=1,ARF=2,PROJ=3,ECAT=4,SINO=5,ACCEL=6,LMF=7,CT=8,DAQ=9,MDB=10,SIZE=11,
	// GPUSPECT=12}
	char SplitNumberAsString[10];
	sprintf(SplitNumberAsString,"%i",splitNumber);

        if( TreatOutputStream("/gate/output/root/setFileName", "rootfile", originalRootFileName, SplitNumberAsString) )
        {
                AddPWD("/gate/output/root/setFileName");
                splitfile<<"Root filename: "<<macline.substr(30,macline.length())<<endl;
                filenames[ROOT]=1;
        }
        else if( TreatOutputStream("/gate/output/ascii/setFileName", "asciifile", originalAsciiFileName, SplitNumberAsString) )
        {
                AddPWD("/gate/output/ascii/setFileName");
                filenames[ASCII]=1;
        }
        else if( TreatOutputStream("/gate/output/arf/setFileName", "arffile", originalARFFileName, SplitNumberAsString) )
        {
                AddPWD("/gate/output/arf/setFileName");
                filenames[ARF]=1;
        }
        else if (TreatOutputStream("/gate/output/projection/setFileName", "projfile", originalProjFileName, SplitNumberAsString))
        {
                AddPWD("/gate/output/projection/setFileName");
                filenames[PROJ]=1;
        }
	else if( TreatOutputStream("/gate/output/ecat7/setFileName", "ecat7file", originalEcat7FileName, SplitNumberAsString) )
	{
		AddPWD("/gate/output/ecat7/setFileName");
		filenames[ECAT]=1;
	}
        else if( TreatOutputStream("/gate/output/sinogram/setFileName", "sinogramfile", originalSinoFileName, SplitNumberAsString) )
        {
                AddPWD("/gate/output/sinogram/setFileName");
                filenames[SINO]=1;
        }
	else if( TreatOutputStream("/gate/output/sinoAccel/setFileName", "accelfile", originalAccelFileName, SplitNumberAsString) )
        {
                AddPWD("/gate/output/sinoAccel/setFileName");
                filenames[ACCEL]=1;
        }
	else if( TreatOutputStream("/gate/output/lmf/setFileName", "lmffile", originalLmfFileName, SplitNumberAsString) )
	{
		AddPWD("/gate/output/lmf/setFileName");
		filenames[LMF]=1;
	}
	else if( TreatOutputStream("/gate/output/imageCT/setFileName", "ctfile", originalCTFileName, SplitNumberAsString) )
        {
                AddPWD("/gate/output/imageCT/setFileName");
                filenames[CT]=1;
        }
	else if( TreatOutputStream("/gate/output/spectGPU/setFileName", "spectGPUfile", originalSPECTGPUFileName, SplitNumberAsString) )
        {
                AddPWD("/gate/output/spectGPU/setFileName");
                filenames[GPUSPECT]=1;
        }
	else if( macline.contains("/gate/geometry/setMaterialDatabase") )
	{
		AddPWD("/gate/geometry/setMaterialDatabase");
		enable[MDB]=1;
	}
}

void GateMacfileParser::SearchForActors(G4int splitNumber,ofstream& output, ofstream& splitfile)
{
  if (macline.contains("/gate/actor/addActor"))
  {
    G4int pos = macline.find_first_of(" ");
    G4String tmp = macline.substr(pos+1);
    pos = tmp.find_first_of(" ");
    G4String actorType = tmp.substr(0,pos);
    G4String actorName = tmp.substr(pos+1);
    listOfActorType.push_back(actorType);
    listOfActorName.push_back(actorName);
  }
  else if (macline.contains("/gate/actor") && (macline.contains("/save ") || macline.contains("/save\t")))
  {
    // We get first the name of the actor in the macro command
    G4String tmp = macline;
    tmp.erase(0,12);
    G4int pos_slash = tmp.find_first_of("/");
    G4String actorName = tmp.substr(0,pos_slash);
    // Then we search if this actor has previously been detected in using the addActor command
    bool findInList = false;
    for (size_t i=0; i<listOfActorName.size(); i++)
    {
      if (actorName == listOfActorName[i])
      {
        listOfEnabledActorName.push_back(actorName);
        listOfEnabledActorType.push_back(listOfActorType[i]);
	listOfActorName.erase(listOfActorName.begin()+i);
	listOfActorType.erase(listOfActorType.begin()+i);
        findInList=true;
        break;
      }
    }
    // If it is the case we registered this actor as enabled and we split its filename
    if (findInList)
    {
      AddSplitNumberWithExtension(splitNumber);
      AddPWD("/gate/actor/"+actorName+"/save");
    }
    // Else, it is an error, this actor does not exist !
    else
    {
      cerr << "Found the 'save' command on an actor that has not previously been declared !" << endl;
      CleanAbort(output,splitfile);
      exit(1);
    }
  }

}

void GateMacfileParser::AddSplitNumberWithExtension(G4int splitNumber)
{
  G4int pos_dot = macline.find_last_of(".");
  G4String extension = macline.substr(pos_dot);
  G4String macline_without_ext = macline.substr(0,pos_dot);
  char tmp[10];
  sprintf(tmp,"%d",splitNumber);
  macline = macline_without_ext+((string)tmp)+extension;
}

bool GateMacfileParser::IsComment(G4String line)
{
	G4String subString=line(0);
	G4int i=0;
	while (subString.contains(" "))
	{
		i++;
		subString=line(i);
	}
	if (subString.contains("#") || line.empty()) return true;
	else return false;
}

void GateMacfileParser::FormatMacline()
{
	G4int size=0;
	G4String subString;
	G4int position=0;
	//remove any trailing comments
	if (macline.contains("#"))
	{
		position=macline.find("#");
		G4String temp=macline.substr(0,position);
		macline=temp;
	}

	//remove "\t"
        while (macline.contains("\t"))
        {
               position = macline.find("\t");
               macline.replace(position,1," ");
        }
	
	//remove trailing spaces
	size=macline.length();
	subString=macline(size-1);
	while (subString.contains(" "))
	{
		size--;
		subString=macline(size);
	}
	if (size<(int)macline.length())
	{
		subString=macline.substr(0,size+1);
		macline=subString;
	}

	//remove starting spaces
	subString=macline(0);
	size=0;
	while (subString.contains(" "))
	{
		size++;
		subString=macline(size);
	}
	if (size>0)
	{
		subString=macline.substr(size,macline.length());
		macline=subString;
	} 

	//remove intermediate spaces >1
	subString=macline;
	while (macline.contains("  "))
	{
		position=macline.find("  ");
		subString=macline.substr(position+1,macline.length());
		macline.replace(position,macline.length(),subString);
	}
}

bool GateMacfileParser::Braced(G4String origFile)
{
	//check for braces in name i.e. unresolved alias
	if(origFile.find("{")!=string::npos&&origFile.find("}")!=string::npos) return true;
	return false;
}

void GateMacfileParser::BraceReplace(G4String def, G4String origFile, char* SplitNumberAsString)
{
	//replace the unresolved alias with default
	if(SplitNumberAsString[0]=='1'&&m_verboseLevel>0)
		cout<<"Warning : alias expected for "<< origFile<<" using default name "<< def <<" and local output !"<<endl;
	macline.replace(macline.length()-origFile.length(),macline.length(),def);
}

const G4String GateMacfileParser::ExtractFileName(G4String key)
{
	//int pos=macline.find(key)+key.length();
	// ugly, is there a simple way to get rid of blanks?
	//char* a=const_cast<char*>( (macline.substr(pos,macline.length())).c_str());
	//return strtok(a," ");
	//this is a simple way ;)
	G4String temp=macline;
	temp.erase(0,key.length()+1);
	return temp;
}

bool GateMacfileParser::TreatOutputStream(G4String key, G4String def, G4String& origFile, char* SplitNumberAsString)
{
	if (!macline.contains(key)) return false; // nothing to do
	origFile = ExtractFileName(key); // token after key
	// look for possible unresolved aliases and replace them with default name
	if( Braced(origFile) )
	{
		BraceReplace(def,origFile,SplitNumberAsString);
		origFile=def;
	}
	//add split number
	macline+=SplitNumberAsString;//NOT NEEDED +Extension;
	return true;
}

/// Misc functions
void GateMacfileParser::skipComment(istream & is)
{
  char c;
  char line[1024];
  if (is.eof()) return;
  is >> c;
  while (is && (c == '#')) {
    is.getline (line, 1024);
    is >> c;
    if (is.eof()) return;
  }
  is.unget();
}

bool GateMacfileParser::ReadColNameAndUnit(istream & is, string name, string & unit) {
  skipComment(is);
  // Read name
  string s;
  is >> s;
  if (s != name) {
    for(unsigned int i=0; i<s.size(); i++) is.unget();
    return false;
  }
  // Read unit name and convert
  is >> unit;
  return true;
}


