//
// Created by mdupont on 12/03/18.
//

#pragma once

#include <functional>
#include <vector>
#include <string>
#include <map>

#include "GateTreeFile.hh"


typedef const std::function<std::unique_ptr<GateOutputTreeFile>()> TCreateOutputTreeFileMethod;
typedef std::map<const std::string,TCreateOutputTreeFileMethod> CreateOutputTreeFileMethodMap;



class GateOutputTreeFileFactory
{
public:
  static bool _register(const std::string name, TCreateOutputTreeFileMethod& funcCreate);
  static std::unique_ptr<GateOutputTreeFile> _create(const std::string& name);

private:
  GateOutputTreeFileFactory() = delete;
  static CreateOutputTreeFileMethodMap& get_methods_map()
  {
    static CreateOutputTreeFileMethodMap s_methods;
    return s_methods;
  }

};



class GateOutputTreeFileManager
{
public:
  GateOutputTreeFileManager();
  GateOutputTreeFileManager(GateOutputTreeFileManager &&m);
  virtual ~GateOutputTreeFileManager();

  std::unique_ptr<GateOutputTreeFile> add_file(const std::string &file_path, const std::string &kind);

  template<typename T>
  void write_variable(const std::string &name, const T *p)
  {
    for(auto&& f : m_listOfTreeFile)
    {
      f->write_variable(name, p, typeid(T));
    }
  }

  void write_variable(const std::string &name, const std::string *p, size_t nb_char);
  void write_variable(const std::string &name, const char *p, size_t nb_char);
  void write_variable(const std::string &name, const int *p, size_t sizeArray);
  void set_tree_name(const std::string &name);

  void fill();
  void close();
  void write_header();
  void write();



private:
  std::vector<std::unique_ptr<GateOutputTreeFile>> m_listOfTreeFile;
  std::string m_nameOfTree;
};



typedef const std::function<std::unique_ptr<GateInputTreeFile>()> TCreateInputTreeFileMethod;
typedef std::map<const std::string,TCreateInputTreeFileMethod> CreateInTreeFileMethodMap;


class GateInputTreeFileFactory
{
public:
  static bool _register(const std::string name, TCreateInputTreeFileMethod& funcCreate);
  static std::unique_ptr<GateInputTreeFile> _create(const std::string& name);



private:
  GateInputTreeFileFactory() = delete;
  static CreateInTreeFileMethodMap& get_methods_map()
  {
    static CreateInTreeFileMethodMap s_methods;
    return s_methods;
  }

};

class GateInputTreeFileManager
{
public:

//  static std::unique_ptr<InputTreeFile> get_file(const std::string &file_path, const std::string &kind);

  GateInputTreeFileManager();
  std::unique_ptr<GateInputTreeFile> set_file(const std::string &file_path, const std::string &kind);
  bool data_to_read();
  void close();
  void read_next_entrie();
  void read_entrie(const uint64_t& i);
  void read_header();
  void set_tree_name(const std::string &name);
  uint64_t nb_elements();

  template<typename T>
  void read_variable(const std::string name, T *p)
  {
    read_variable(name, p, typeid(T));
  }

  void read_variable(const std::string &name, char* p);
  void read_variable(const std::string &name, std::string *p);
  void read_variable(const std::string &name, char* p, size_t nb_char);

  bool has_variable(const std::string &name);
  std::type_index get_type_of_variable(const std::string &name);



private:
  void read_variable(const std::string &name, void *p, std::type_index t_index);
  std::unique_ptr<GateInputTreeFile> m_inputtreefile;
  std::string m_nameOfTree;
};


class GateInputTreeFileChain
{

public:
  std::unique_ptr<GateInputTreeFile> add_file(const std::string &file_path, const std::string &kind);
  bool data_to_read();
  void read_entrie();
  void read_header();
  void close();

  GateInputTreeFileChain();

  void set_tree_name(const std::string &name);
  uint64_t nb_elements();
  void read_entrie(const uint64_t& i);

  template<typename T>
  void read_variable(const std::string name, T *p)
  {
    read_variable(name, p, typeid(T));
  }

  void read_variable(const std::string &name, char* p);
  void read_variable(const std::string &name, std::string *p);
  void read_variable(const std::string &name, char* p, size_t nb_char);

  bool has_variable(const std::string &name);
  std::type_index get_type_of_variable(const std::string &name);


private:
  void read_variable(const std::string &name, void *p, std::type_index t_index);

  std::vector<std::unique_ptr<GateInputTreeFile>> m_listOfTreeFile;
  std::string m_nameOfTree;

};





