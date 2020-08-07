//
// Created by mdupont on 13/03/18.
//

#include "GateAsciiFile.hh"

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cstring> //for strerror
#include <sstream>

#include"GateMessageManager.hh"


#include "GateFileExceptions.hh"

using namespace std;

void GateAsciiTree::register_variable(const std::string &name, const void *p, std::type_index t_index)
{
   // std::cout << "AsciiFile::register_variable name = " << name << " t_index = " << t_index.name() << endl;
   //  AsciiData d(p, name, t_index, m_tmapOfSave_to_file.at(t_index));
    GateAsciiData dd(p, name, t_index, m_tmapOfSave_to_file.at(t_index), m_tmapOfRead_from_string.at(t_index));

    for (auto&& d : m_vector_of_pointer_to_data) // access by const reference
    {
        if(d.name() == name)
        {
            string s("Error: Key '");
            s+= name;
            s += "' already used !";
            cerr << s << endl;
            throw GateKeyAlreadyExistsException( s );
        }
    }

    m_vector_of_pointer_to_data.push_back(dd);

}

void GateAsciiTree::register_variable(const std::string &name, const char *p, size_t )
{
    this->register_variable(name, p, typeid(char*));
}

void GateAsciiTree::register_variable(const std::string &name, const std::string *p, size_t )
{
    this->register_variable(name, p, typeid(string));
}

void  GateAsciiTree::register_variable(const std::string &name, const int *, size_t )
{
    //problem unordered map at key
    //this->register_variable(name, p, typeid(int*));
    std::cout<<"ERROR "<<name<<" information can not be written in ascii output file"<<std::endl;
    GateError(" .txt format  is not available for *int output such as  volumeID. Unselect volumeID information in your output file or select .root output file");

}


bool GateOutputAsciiTreeFile::is_open()
{
  return m_file.is_open();
}

bool GateInputAsciiTreeFile::is_open()
{
  return m_file.is_open();
}



void GateOutputAsciiTreeFile::fill()
{
    uint32_t i = 0;
    for (auto&& d : m_vector_of_pointer_to_data) // access by const reference
    {

        if (d.m_type_index == typeid(char*) )
        {
            const char* p = (const  char *)d.m_pointer_to_data;
            m_file << p;
        }
        else
        {
            d.m_save_to_file(&d, m_file);
        }
        if (i < m_vector_of_pointer_to_data.size() - 1)
            m_file << m_sep;
        i++;
    }
    m_file << "\n";
}


GateAsciiTree::GateAsciiTree() : m_sep(",")
{
    this->attach_read_and_write_function<double>();
    this->attach_read_and_write_function<float>();
    this->attach_read_and_write_function<uint8_t>();
    this->attach_read_and_write_function<uint16_t>();
    this->attach_read_and_write_function<uint32_t>();
    this->attach_read_and_write_function<uint64_t>();

    this->attach_read_and_write_function<int8_t>();
    this->attach_read_and_write_function<int16_t>();
    this->attach_read_and_write_function<int32_t>();
    this->attach_read_and_write_function<int64_t>();

    this->attach_read_and_write_function<bool>();
    this->attach_read_and_write_function<char>();

    this->attach_read_and_write_function<string>();
    this->attach_read_and_write_function<char*>();
}


void GateOutputAsciiTreeFile::write_header()
{
    if(!m_file.is_open())
        throw GateClosedFileException("InputAsciiTreeFile::write_header: try to write in closed file");

    if( (m_mode & ios_base::out) != ios_base::out )
        throw GateModeFileException("OutputAsciiTreeFile::write_header: file not opened in write mode"); // Should not happen

    m_file << "#";
    uint32_t i = 0;
    for (auto&& d : m_vector_of_pointer_to_data) // access by const reference
    {
        m_file << d.name();
        if (i < m_vector_of_pointer_to_data.size() - 1)
            m_file << m_sep;
        i++;
    }
    m_file << "\n";
}

GateAsciiData::GateAsciiData(const void *pointer_to_data,
                     const string &name,
                     const type_index &type_index,
                     save_to_file_f &save_to_file, read_from_string_f &read_from_string) :
    GateData(pointer_to_data,
         name,
         type_index),
    m_save_to_file(save_to_file),
    m_read_from_string(read_from_string),
    m_index_of_this_data_in_header(0),
    m_max_caracter_accepted_by_provided_read_buffer(0)
{}

template<>
void read_from_string<char*>(GateAsciiData *d, const std::string& s)
{
//  std::cout << "read to char* (name = " << d->name() << ")" << "m_max_caracter_accepted_by_provided_read_buffer = " << d->m_max_caracter_accepted_by_provided_read_buffer << std::endl;
  if(d->m_max_caracter_accepted_by_provided_read_buffer)
    strncpy((char*)d->m_pointer_to_data, s.c_str(), d->m_max_caracter_accepted_by_provided_read_buffer);
  else
    strcpy((char*)d->m_pointer_to_data, s.c_str()); //dangerous :'( but we don't know the size of m_pointer_to_data here
}


void GateOutputAsciiTreeFile::write_variable(const std::string &name, const void *p, std::type_index t_index)
{
    this->register_variable(name, p, t_index);
}

void GateOutputAsciiTreeFile::write_variable(const std::string &name, const std::string *p, size_t nb_char)
{
    this->register_variable(name, p, nb_char);
}

void GateOutputAsciiTreeFile::write_variable(const std::string &name, const char *p, size_t nb_char)
{
    this->register_variable(name, p, nb_char);
}

void GateOutputAsciiTreeFile::write_variable(const std::string &name, const int *p, size_t n)
{
    this->register_variable(name, p, n);
}

void GateOutputAsciiTreeFile::open(const std::string& s)
{
  GateFile::open(s, std::fstream::out);
  m_file.open(s, std::fstream::out);

  if(!m_file.is_open())
  {
    std::stringstream ss;
    ss << "Error opening file! '"  << s <<  "' : " << strerror(errno) ;
    throw std::ios::failure(ss.str());
  }

  m_file << std::setprecision(10);
}

void GateOutputAsciiTreeFile::write()
{ }

void GateOutputAsciiTreeFile::close()
{
  m_file.close();
}

void GateInputAsciiTreeFile::close()
{
  m_file.close();
}


void GateInputAsciiTreeFile::open(const std::string &s)
{
  GateFile::open(s, std::fstream::in);
  m_file.open(s, std::fstream::in);
  m_file << std::setprecision(10);

  if(!m_file.is_open())
  {
    std::stringstream ss;
    ss << "Error opening file! '"  << s <<  "' : " << strerror(errno) ;
    throw std::ios::failure(ss.str());
  }

  m_file.clear();
  m_file.seekg (0, fstream::beg);

    std::string line;
    while (std::getline(m_file, line))
      ++m_number_of_lines_in_file;
    m_file.clear();
    m_file.seekg (0, fstream::beg);

    m_number_of_lines_in_file--; //skip header
}


vector<string> split_string(const string &line, const string &sep)
{
    std::string s = line;
    vector<string> ret;

    if(line.empty())
      return ret;

    if(s.find(sep) == std::string::npos)
    {
      // Only one element, return it
      ret.push_back(s);
      return ret;
    }

    size_t pos = 0;
    std::string token;
    while ((pos = s.find(sep)) != std::string::npos) {
        token = s.substr(0, pos);
//        cout << "token = " << token << " ";
        ret.push_back(token);
        s.erase(0, pos + sep.length());
    }

    ret.push_back(s);


//    cout << endl;
    return ret;
}


void GateInputAsciiTreeFile::read_header()
{
    if(!is_open())
        throw GateClosedFileException("InputAsciiTreeFile::read_header: try to read from closed file");

    string                line;
    getline(m_file,line);

    size_t pos_start = line.find_last_of('#');

    if (pos_start != string::npos)
        line = line.substr(pos_start + 1);
    else
        throw GateMissingHeaderException("InputAsciiTreeFile::read_header: no header found in file");

    m_list_from_header = split_string(line, m_sep);

    if(m_list_from_header.empty())
        throw GateMalFormedHeaderException("InputAsciiTreeFile::read_header: empty header found in file");

    m_start_of_data = m_file.tellg();
    m_read_header_called = true;

}


void GateInputAsciiTreeFile::read_next_entrie()
{
    string line;

    getline(m_file,line);

//    cout << "line read = " << line << endl;
    auto vs = split_string(line, m_sep);

    for (auto&& d : m_vector_of_pointer_to_data){
//        cout << "search variable = " << d.name() <<  " index = " <<  d.m_index_of_this_data_in_header <<   endl;
        string s = vs.at(d.m_index_of_this_data_in_header);
        d.m_read_from_string(&d, s);
    }

  m_nb_entries_read++;

}
void GateInputAsciiTreeFile::read_variable(const std::string &name, void *p, std::type_index t_index)
{
    if(!m_read_header_called)
        throw std::logic_error("read_header not called");

    bool found = false;

    for(size_t i = 0; i < m_list_from_header.size(); ++i)
    {
        auto name_from_header  = m_list_from_header[i];
        if (name_from_header == name)
        {
            register_variable(name, p, t_index);
            auto d = m_vector_of_pointer_to_data.back();
            m_vector_of_pointer_to_data.pop_back();
            d.m_index_of_this_data_in_header = i;
            m_vector_of_pointer_to_data.push_back(d);
            found = true;
        }
    }

    if(!found)
    {
      std::stringstream ss;
      ss << "Variable named '" << name << "' not found !";
      throw GateKeyNotFoundInHeaderException(ss.str());
    }


}
bool GateInputAsciiTreeFile::data_to_read()
{
  if(!m_read_header_called)
    throw std::logic_error("read_header not called");

  return m_nb_entries_read < m_number_of_lines_in_file;
}

void GateInputAsciiTreeFile::read_variable(const std::string &name, std::string *p)
{
  this->read_variable(name, p, typeid(string));
}

void GateInputAsciiTreeFile::read_variable(const std::string &name, char *p)
{
  this->read_variable(name, p, typeid(char*));
}

void GateInputAsciiTreeFile::read_variable(const std::string &name, char *p, size_t nb_char)
{
//  cout << "read_variable = " << nb_char << endl;
  this->read_variable(name, p, typeid(char*));

  for (auto&& d : m_vector_of_pointer_to_data){
    if(d.name() ==name )
    {
      d.m_max_caracter_accepted_by_provided_read_buffer = nb_char;
    }
  }
}

GateInputAsciiTreeFile::GateInputAsciiTreeFile():
  m_number_of_lines_in_file(0),
  m_nb_entries_read(0),
  m_read_header_called(false)
{

}


bool GateInputAsciiTreeFile::has_variable(const std::string &name)
{
  if(!m_read_header_called)
    throw std::logic_error("read_header not called");
  for (auto&& d : m_list_from_header){
    if(d ==name )
    {
      return true;
    }
  }
  return false;
}

type_index GateInputAsciiTreeFile::get_type_of_variable(const std::string & /*name*/)
{
  if(!m_read_header_called)
    throw std::logic_error("read_header not called");
//  return typeid(nullptr);
  throw GateNoTypeInHeaderException("");
}

uint64_t GateInputAsciiTreeFile::nb_elements()
{
  return m_number_of_lines_in_file;
}

void GateInputAsciiTreeFile::read_entrie(const uint64_t &i)
{
  m_file.seekg(m_start_of_data);

  for(uint64_t nb_seek = 0; nb_seek < i; ++nb_seek)
  {
    string line;
    getline(m_file, line);
  }

  this->read_next_entrie();

}

bool GateOutputAsciiTreeFile::s_registered =  GateOutputTreeFileFactory::_register(GateOutputAsciiTreeFile::_get_factory_name(), &GateOutputAsciiTreeFile::_create_method<GateOutputAsciiTreeFile>);
bool GateInputAsciiTreeFile::s_registered =  GateInputTreeFileFactory::_register(GateInputAsciiTreeFile::_get_factory_name(), &GateInputAsciiTreeFile::_create_method<GateInputAsciiTreeFile>);













