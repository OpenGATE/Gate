# harvard.perl
# Copyright 1994 - thorsten schnier (thorsten@arch.su.edu.au)
# key center of design computing, university of sydney
#
# This work may be distributed and/or modified under the
# conditions of the LaTeX Project Public License, either version 1.3
# of this license or (at your option) any later version.
# The latest version of this license is in
#   http://www.latex-project.org/lppl.txt
# and version 1.3 or later is part of all distributions of LaTeX
# version 2005/12/01 or later.
#
# This work has the LPPL maintenance status `maintained'.
# 
# The Current Maintainers of this work are Peter Williams and Thorsten Schnier.
#
# This work consists of all files listed in manifest.txt.
#
# Licence notice added on behalf of Peter Williams and Thorsten Schnier
# by Clea F. Rees 2009/01/30.
#
# Change Log:
# ===========
#
# 07-JUN-94 first attempt
# 16-JUN-94 changes for harvard 2.0
#           (star and starstar commands)


package main;

$cite_mark = '<tex2html_cite_mark>';
$citefull_mark = '<tex2html_citefull_mark>';
$citeshort_mark = '<tex2html_citeshort_mark>';
$citename_mark = '<tex2html_citename_mark>';
$citenamefull_mark = '<tex2html_citenamefull_mark>';
$citenameshort_mark = '<tex2html_citenameshort_mark>';
$citekey_mark = '<tex2html_citekey_mark>';
$citeyear_mark = '<tex2html_citeyear_mark>';

$harvard_left = '(';
$harvard_right = ')';
$harvardyear_left = '(';
$harvardyear_right = ')';
$harvard_cite_separator = $harvard_agsm_cite_separator = ', ';
$harvard_year_separator = $harvard_agsm_year_separator = " ";
$harvard_agsm_and = '&amp;';
$harvard_dcu_cite_separator = "; ";
$harvard_dcu_year_separator = ", ";
$harvard_dcu_and = 'and';
$harvard_and = $harvard_agsm_and;
$harvard_default = 1;
$harvard_full = 0;

# the citation commands
# This just creates a link from a label (yet to be determined) to the 
# cite_key in the citation file.
# use different markers, depending on the style of the citation used

#changes to latex2html-version:
#insert separating sign, use optional text argument, use brackets 
#according to cite-mode, and reverse order of cites (to correct order)



sub do_cmd_cite {
    local($_) = @_;
    local($text, $dummy) = &get_next_optional_argument;
    s/$next_pair_pr_rx//o;
    local($cite_key) = $2;
    local ($first) = 1 ;
    local ($result);
    if ($cite_key) {
	foreach $cite_key (split(/,/,$cite_key)) {
	    $cite_key =~ s/\W//g;	  # Remove non alphanumeric characters
	    $result = join ('', $result,
			    $first ? $harvard_left : $harvard_cite_separator,
			    "<A HREF=#$cite_key>$citekey_mark", 
			    '##', $cite_key, $cite_mark, 
			    $harvard_year_separator,
			    '##', $cite_key, $citeyear_mark,
			    "<\/A>");
	    $first = 0;
	}
	join ('', 
	      $result,
	      $text ? ($harvard_agsm_cite_separator, $text):'', 
	      $harvard_right,
	      $_);
    }
    else {warn "Cannot find citation argument\n";}
}


sub do_cmd_citestar {
    local($_) = @_;
    local($text, $dummy) = &get_next_optional_argument;
    s/$next_pair_pr_rx//o;
    local($cite_key) = $2;
    local ($first) = 1 ;
    local ($result);
    if ($cite_key) {
	foreach $cite_key (split(/,/,$cite_key)) {
	    $cite_key =~ s/\W//g;	  # Remove non alphanumeric characters
	    $result = join ('', $result,
			    $first ? $harvard_left : $harvard_cite_separator,
			    "<A HREF=#$cite_key>$citekey_mark", 
			    '##', $cite_key, $citefull_mark, 
			    $harvard_year_separator,
			    '##', $cite_key, $citeyear_mark,
			    "<\/A>");
	    $first = 0;
	}
	join ('', 
	      $result,
	      $text ? ($harvard_agsm_cite_separator, $text):'', 
	      $harvard_right,
	      $_);
    }
    else {warn "Cannot find citation argument\n";}
}


sub do_cmd_citestarstar {
    local($_) = @_;
    local($text, $dummy) = &get_next_optional_argument;
    s/$next_pair_pr_rx//o;
    local($cite_key) = $2;
    local ($first) = 1 ;
    local ($result);
    if ($cite_key) {
	foreach $cite_key (split(/,/,$cite_key)) {
	    $cite_key =~ s/\W//g;	  # Remove non alphanumeric characters
	    $result = join ('', $result,
			    $first ? $harvard_left : $harvard_cite_separator,
			    "<A HREF=#$cite_key>$citekey_mark", 
			    '##', $cite_key, $citeshort_mark, 
			    $harvard_year_separator,
			    '##', $cite_key, $citeyear_mark,
			    "<\/A>");
	    $first = 0;
	}
	join ('', 
	      $result,
	      $text ? ($harvard_agsm_cite_separator, $text):'', 
	      $harvard_right,
	      $_);
    }
    else {warn "Cannot find citation argument\n";}
}



sub do_cmd_citeaffixed {
    local($_) = @_;
    local($text, $dummy) = &get_next_optional_argument;
    s/$next_pair_pr_rx//o;
    local($cite_key) = $2;
    s/$next_pair_pr_rx//o;
    local($comment) = $2;
    local ($first) = 1;
    local ($result);
     if ($cite_key) {
	foreach $cite_key (split(/,/,$cite_key)) {
	    $cite_key =~ s/\W//g;	  # Remove non alphanumeric characters
	    $result = join ('', $result, 
			    $first? ($harvard_left, $comment, " ")
			             : $harvard_cite_separator,
			    "<A HREF=#$cite_key>$citekey_mark", 
			    '##', $cite_key, $cite_mark, 
			    $harvard_year_separator,
			    '##', $cite_key,$citeyear_mark, 
			    "<\/A>");
	    $first = 0;
	}
	join('',
	     $result, 
	     $text? ($harvard_agsm_cite_separator, $text):'', 
	     $harvard_right,
	     $_);
    }
    else {warn "Cannot find citation argument\n";}
}


sub do_cmd_citeaffixedstar {
    local($_) = @_;
    local($text, $dummy) = &get_next_optional_argument;
    s/$next_pair_pr_rx//o;
    local($cite_key) = $2;
    s/$next_pair_pr_rx//o;
    local($comment) = $2;
    local ($first) = 1;
    local ($result);
     if ($cite_key) {
	foreach $cite_key (split(/,/,$cite_key)) {
	    $cite_key =~ s/\W//g;	  # Remove non alphanumeric characters
	    $result = join ('', $result, 
			    $first? ($harvard_left, $comment, " ")
			             : $harvard_cite_separator,
			    "<A HREF=#$cite_key>$citekey_mark", 
			    '##', $cite_key, $citefull_mark, 
			    $harvard_year_separator,
			    '##', $cite_key,$citeyear_mark, 
			    "<\/A>");
	    $first = 0;
	}
	join('',
	     $result, 
	     $text? ($harvard_agsm_cite_separator, $text):'', 
	     $harvard_right,
	     $_);
    }
    else {warn "Cannot find citation argument\n";}
}

sub do_cmd_citeaffixedstarstar {
    local($_) = @_;
    local($text, $dummy) = &get_next_optional_argument;
    s/$next_pair_pr_rx//o;
    local($cite_key) = $2;
    s/$next_pair_pr_rx//o;
    local($comment) = $2;
    local ($first) = 1;
    local ($result);
     if ($cite_key) {
	foreach $cite_key (split(/,/,$cite_key)) {
	    $cite_key =~ s/\W//g;	  # Remove non alphanumeric characters
	    $result = join ('', $result, 
			    $first? ($harvard_left, $comment, " ")
			             : $harvard_cite_separator,
			    "<A HREF=#$cite_key>$citekey_mark", 
			    '##', $cite_key, $citeshort_mark, 
			    $harvard_year_separator,
			    '##', $cite_key,$citeyear_mark, 
			    "<\/A>");
	    $first = 0;
	}
	join('',
	     $result, 
	     $text? ($harvard_agsm_cite_separator, $text):'', 
	     $harvard_right,
	     $_);
    }
    else {warn "Cannot find citation argument\n";}
}



#similar to cite
#no multiple cites allowed !

sub do_cmd_citeasnoun {
    local($_) = @_;
    local($text, $dummy) = &get_next_optional_argument;
    s/$next_pair_pr_rx//o;
    local($cite_key) = $2;
    if ($cite_key) {
	$cite_key =~ s/\W//g;		# Remove non alphanumeric characters
	join ('',
	      "<A HREF=#$cite_key>$citekey_mark",
	      '##', $cite_key,$cite_mark,
	      ' ',
	      $harvard_left,
	      '##', $cite_key,$citeyear_mark,
	      $text ? ($harvard_agsm_cite_separator, $text):'',
	      $harvard_right,
	      "<\/A>",
	      $_);
    }	
    else {warn "Cannot find citation argument\n";}
}


sub do_cmd_citeasnounstar {
    local($_) = @_;
    local($text, $dummy) = &get_next_optional_argument;
    s/$next_pair_pr_rx//o;
    local($cite_key) = $2;
    if ($cite_key) {
	$cite_key =~ s/\W//g;		# Remove non alphanumeric characters
	join ('',
	      "<A HREF=#$cite_key>$citekey_mark",
	      '##', $cite_key,$citefull_mark,
	      ' ',
	      $harvard_left,
	      '##', $cite_key,$citeyear_mark,
	      $text ? ($harvard_agsm_cite_separator, $text):'',
	      $harvard_right,
	      "<\/A>",
	      $_);
    }	
    else {warn "Cannot find citation argument\n";}
}


sub do_cmd_citeasnounstarstar {
    local($_) = @_;
    local($text, $dummy) = &get_next_optional_argument;
    s/$next_pair_pr_rx//o;
    local($cite_key) = $2;
    if ($cite_key) {
	$cite_key =~ s/\W//g;		# Remove non alphanumeric characters
	join ('',
	      "<A HREF=#$cite_key>$citekey_mark",
	      '##', $cite_key,$citeshort_mark,
	      ' ',
	      $harvard_left,
	      '##', $cite_key,$citeyear_mark,
	      $text ? ($harvard_agsm_cite_separator, $text):'',
	      $harvard_right,
	      "<\/A>",
	      $_);
    }	
    else {warn "Cannot find citation argument\n";}
}



#same as citeasnoun, but add the possessive 's
#no multiple cites allowed !

sub do_cmd_possessivecite {
    local($_) = @_;
    local($text, $dummy) = &get_next_optional_argument;
    s/$next_pair_pr_rx//o;
    local($cite_key) = $2;
    if ($cite_key) {
	$cite_key =~ s/\W//g;		# Remove non alphanumeric characters
	# The proper link $citefile#$cite_key will be substituted later
        join('', "<A HREF=#$cite_key>$citekey_mark",
	     '##', $cite_key,$cite_mark,
	     '\'s ',
	     $harvard_left,
	     '##', $cite_key,$citeyear_mark,
	     $text? ($harvard_agsm_cite_separator, $text):'', 
	     $harvard_right,
	     "<\/A>",
	     $_);
    }
    else {warn "Cannot find citation argument\n";}
}

sub do_cmd_possessivecitestar {
    local($_) = @_;
    local($text, $dummy) = &get_next_optional_argument;
    s/$next_pair_pr_rx//o;
    local($cite_key) = $2;
    if ($cite_key) {
	$cite_key =~ s/\W//g;		# Remove non alphanumeric characters
	# The proper link $citefile#$cite_key will be substituted later
        join('', "<A HREF=#$cite_key>$citekey_mark",
	     '##', $cite_key,$citefull_mark,
	     '\'s ',
	     $harvard_left,
	     '##', $cite_key,$citeyear_mark,
	     $text? ($harvard_agsm_cite_separator, $text):'', 
	     $harvard_right,
	     "<\/A>",
	     $_);
    }
    else {warn "Cannot find citation argument\n";}
}

sub do_cmd_possessivecitestarstar {
    local($_) = @_;
    local($text, $dummy) = &get_next_optional_argument;
    s/$next_pair_pr_rx//o;
    local($cite_key) = $2;
    if ($cite_key) {
	$cite_key =~ s/\W//g;		# Remove non alphanumeric characters
	# The proper link $citefile#$cite_key will be substituted later
        join('', "<A HREF=#$cite_key>$citekey_mark",
	     '##', $cite_key,$citeshort_mark,
	     '\'s ',
	     $harvard_left,
	     '##', $cite_key,$citeyear_mark,
	     $text? ($harvard_agsm_cite_separator, $text):'', 
	     $harvard_right,
	     "<\/A>",
	     $_);
    }
    else {warn "Cannot find citation argument\n";}
}


#similar to citeasnoun, but only names
#multiple arguments allowed, no link produced
#no optional text argument

sub do_cmd_citename {
    local($_) = @_;
    s/$next_pair_pr_rx//o;
    local($cite_key) = $2;
    local ($first) = 1;
    local ($result);
    if ($cite_key) {
	foreach $cite_key (split(/,/,$cite_key)) {
	    $cite_key =~ s/\W//g;	  # Remove non alphanumeric characters
	    # The proper link $citefile#$cite_key will be substituted later
            $result = join('', $result, 
			   ($first? '': $harvard_cite_separator),
			   "<A HREF=#$cite_key>$citekey_mark",
			   '##', $cite_key, $citename_mark, 
			   "</A>");
	    $first = 0;
	}
	join ('', $result, $_);
    }
    else {warn "Cannot find citation argument\n";}
}

sub do_cmd_citenamestar {
    local($_) = @_;
    s/$next_pair_pr_rx//o;
    local($cite_key) = $2;
    local ($first) = 1 ;
    local ($result);
    if ($cite_key) {
	foreach $cite_key (split(/,/,$cite_key)) {
	    $cite_key =~ s/\W//g;	  # Remove non alphanumeric characters
	    # The proper link $citefile#$cite_key will be substituted later
            $result = join('', $result,
			   ($first? '' : $harvard_cite_separator),
			   "<A HREF=#$cite_key>$citekey_mark",
			   '##', $cite_key,$citenamefull_mark, 
			   "</A>");
	    $first = 0;
	}
	join ('', $result, $_);
    }
    else {warn "Cannot find citation argument\n";}
}

sub do_cmd_citenamestarstar {
    local($_) = @_;
    s/$next_pair_pr_rx//o;
    local($cite_key) = $2;
    local ($first) = 1 ;
    local ($result);
    if ($cite_key) {
	foreach $cite_key (split(/,/,$cite_key)) {
	    $cite_key =~ s/\W//g;	  # Remove non alphanumeric characters
	    # The proper link $citefile#$cite_key will be substituted later
            $result = join('', $result,
			   ($first? '' : $harvard_cite_separator),
			   "<A HREF=#$cite_key>$citekey_mark",
			   '##', $cite_key,$citenameshort_mark, 
			   "</A>");
	    $first = 0;
	}
	join ('', $result, $_);
    }
    else {warn "Cannot find citation argument\n";}
}


#similar to cite, but only year
#multiple arguments allowed, no link produced
#no optional text argument

sub do_cmd_citeyear {
    local($_) = @_;
    s/$next_pair_pr_rx//o;
    local($cite_key) = $2;
    local ($first) = 1 ;
    local ($result);
    if ($cite_key) {
	foreach $cite_key (split(/,/,$cite_key)) {
	    $cite_key =~ s/\W//g;	  # Remove non alphanumeric characters
	    # The proper link $citefile#$cite_key will be substituted later
	    $result = join ('', $result, 
			    ($first ? $harvard_left : $harvard_cite_separator),
			    "<A HREF=#$cite_key>$citekey_mark",
			    '##', $cite_key,$citeyear_mark, 
			    "</A>");
	    $first = 0;
	}
	join('',$result, $harvard_right, $_); # closing bracket
    }
    else {warn "Cannot find citation argument\n";}
}


# citeyear without brackets

sub do_cmd_citeyearstar {
    local($_) = @_;
    s/$next_pair_pr_rx//o;
    local($cite_key) = $2;
    local ($first) = 1 ;
    local ($result);
    if ($cite_key) {
	foreach $cite_key (split(/,/,$cite_key)) {
	    $cite_key =~ s/\W//g;	  # Remove non alphanumeric characters
	    # The proper link $citefile#$cite_key will be substituted later
	    $result = join ('', $result, 
			    ($first ? '' : $harvard_cite_separator),
			    "<A HREF=#$cite_key>$citekey_mark",
			    '##', $cite_key,$citeyear_mark, 
			    "</A>");
	    $first = 0;
	}
	join('',$result, $_); # no brackets
    }
    else {warn "Cannot find citation argument\n";}
}




# the style commands

sub do_cmd_citationstyle{
    local($_) = @_;
    s/$next_pair_pr_rx//o;
    $2;
  switch: {
      if ($2 =~ /dcu/){
	$harvard_cite_separator = $harvard_dcu_cite_separator;
	$harvard_year_separator = $harvard_dcu_year_separator;
	$harvard_and = $harvard_dcu_and;
	last switch; }
      if ($2 =~ /agsm/) {
	$harvard_cite_separator = $harvard_agsm_cite_separator;
	$harvard_year_separator = $harvard_agsm_year_separator;
	$harvard_and = $harvard_agsm_and;
	last switch;}
      warn 'unknown citation style $2';}
    $_;
} 



sub do_cmd_citationmode{
    local($_) = @_;
    s/$next_pair_pr_rx//o;
    $2;
  switch: {
      if ($2 =~ /full/){
	$harvard_full = 1;
	$harvard_default = 0;
	last switch; }
      if ($2 =~ /abbr/) {
	$harvard_full = 0;
	$harvard_default = 0;
	last switch;}
      if ($2 =~ /default/) {
	$harvard_full = 0;
	$harvard_default = 1;
	last switch;}
      warn 'unknown citation mode $2';}
    $_;
} 


sub do_cmd_harvardparenthesis{
    local($_) = @_;
    s/$next_pair_pr_rx//o;
  switch: {
      if ($2 =~ /round/){
	$harvard_left = '\(';
	$harvard_right = '\)';
	$harvardyear_left = '\(';
	$harvardyear_right = '\)';
	last switch; }
      if ($2 =~ /curly/) {
	$harvard_left = '\{';
	$harvard_right = '\}';
	$harvardyear_left = '\{';
	$harvardyear_right = '\}';
	last switch;}
      if ($2 =~ /square/) {
	$harvard_left = '\[';
	$harvard_right = '\]';
	$harvardyear_left = '\[';
	$harvardyear_right = '\]';
	last switch;}
      if ($2 =~ /angle/) {
	$harvard_left = '&lt;';
	$harvard_right = '&gt;';
	$harvardyear_left = '&lt;';
	$harvardyear_right = '&gt;';
	last switch;}
      warn 'unknown paranthesis type $2';}
    $_
} 


sub do_cmd_harvardyearparenthesis{
    local($_) = @_;
    s/$next_pair_pr_rx//o;
  switch: {
      if ($2 =~ /round/){
	$harvardyear_left = '\(';
	$harvardyear_right = '\)';
	last switch; }
      if ($2 =~ /curly/) {
	$harvardyear_left = '\{';
	$harvardyear_right = '\}';
	last switch;}
      if ($2 =~ /square/) {
	$harvardyear_left = '\[';
	$harvardyear_right = '\]';
	last switch;}
      if ($2 =~ /angle/) {
	$harvardyear_left = '&lt;';
	$harvardyear_right = '&gt;';
	last switch;}
      warn 'unknown year paranthesis type $_';}
    $_;
} 


# harvard symbols

sub do_cmd_harvardleft {
    join('',$harvard_left,@_[0]);
}

sub do_cmd_harvardright {
    join('',$harvard_right,@_[0]);
}

sub do_cmd_harvardyearleft {
    join('',$harvardyear_left,@_[0]);
}

sub do_cmd_harvardyearright {
    join('',$harvardyear_right,@_[0]);
}

sub do_cmd_harvardand {
    join('',$harvard_and,@_[0]);
}



# This command will only be encountered inside a thebibliography environment.
# this should not occur in a file using harvard style
# mainly unchanged, only a warning added

sub do_cmd_bibitem {
    local($_) = @_;
    # The square brackets may contain the label to be printed
    warn "bibitem-command in harvard-style file\n Cannot extract all neccessary information for cites\n";
    local($label, $dummy) = &get_next_optional_argument;
    $label = ++$bibitem_counter unless $label; # Numerical labels 
    s/$next_pair_pr_rx//o;
    $cite_key = $2;
    if ($cite_key) {
	$cite_key =~ s/\W//g;		# Remove non alphanumeric characters
	# Associate the cite_key with the printed label.
	# The printed label will be substituted back into the document later.
	$cite_info{$cite_key} = $label;
	# Create an anchor around the citation
	join('',"<DT><A NAME=\"$cite_key\"><B>$label</B></A><DD>", $_);
    }
    else {
	warn "Cannot find bibitem labels\n";}
}

# collect information from item and associate it with the
# cite-key. Contrary to bibitem, the cite-key argument is not optional,
# so no numeric cite-key has to be generated

sub do_cmd_harvarditem {
    local($_) = @_;
    # The square brackets may contain the short citation
    local($abbr_cite, $dummy) = &get_next_optional_argument;
    s/$next_pair_pr_rx//o;
    local($full_cite)= $2;
    $abbr_cite= $full_cite unless $abbr_cite;
    s/$next_pair_pr_rx//o;
    local($cite_year) = $2;
    s/$next_pair_pr_rx//o;
    local($cite_key) = $2;
    if ($cite_key) {
	$cite_key =~ s/\W//g;		# Remove non alphanumeric characters
	# Associate the citation-infos with the cite-key.
	# The infos will be substituted back into the document later,
	# depending on the citation used
	$cite_info_abbr {$cite_key} = $abbr_cite;
	$cite_info_full {$cite_key} = $full_cite;
	$cite_info_year {$cite_key} = $cite_year;
	$cite_info_first {$cite_key} = 1;
	# Create an anchor around the citation
	join('',"<DT><A NAME=\"$cite_key\">- </A><DD>", $_);
    }
    else {
	warn "Cannot find bibitem labels\n";}
}


##
## to allow bibcard to produce URL links:
## same as htmladdnormallink, but only one argument (both link and text)

sub do_cmd_harvardurl{
    local($_) = @_;
    local($url);
    s/$next_pair_pr_rx/$url = $2; ''/eo;
    join('',do {qq/<B>URL:<\/B><A HREF="$url">$url<\/A>/;},$_);
}



# this onle has to be extended to search for citename_mark and
# citeyear_mark as well
# the appropriate entries should have been produced by do_cmd_harvarditem

sub replace_cite_references {
   s/##(\w+)$cite_mark/&help_replace_cites(\1)/ge;
   s/##(\w+)$citefull_mark/&help_replace_fullcites(\1)/ge;
   s/##(\w+)$citeshort_mark/&help_replace_shortcites(\1)/ge;
   s/##(\w+)$citename_mark/&help_replace_name($1)/ge;
   s/##(\w+)$citenamefull_mark/$cite_info_full{\1}/g;
   s/##(\w+)$citenameshort_mark/$cite_info_short{\1}/g;
   s/##(\w+)$citeyear_mark/$cite_info_year{\1}/g;
   s/#(\w+)>$citekey_mark/$citefile#\1>/g;
   $_
}


#return full or abbreviated cite text
sub help_replace_cites {
    local ($key)= @_[0];
    $cite_info_first{$key}= 0;	# not the first time anymore
    if ($harvard_full || ($cite_info_first{$key} && $harvard_default)) {
	$cite_info_full{$key}; # full citation
    }
    else{
	$cite_info_abbr{$key}; # abbreviated citation
    }
}  

sub help_replace_shortcites {
    local ($key)= @_[0];
    $cite_info_first{$key}= 0;	# not the first time anymore
    $cite_info_abbr{$key}; # abbreviated citation
}  

sub help_replace_fullcites {
    local ($key)= @_[0];
    $cite_info_first{$key}= 0;	# not the first time anymore
    $cite_info_full{$key}; # full citation
}  

sub help_replace_name {
    local ($key)= @_[0];
    if ($harvard_full || ($cite_info_first{$key} && $harvard_default)) {
	$cite_info_full{$key}; # full citation
    }
    else{
	$cite_info_abbr{$key}; # abbreviated citation
    }
}


#this one has to be changes as well, so that it finds all citemarks

sub remove_general_markers {
    s/$lof_mark/<UL>$figure_captions<\/UL>/o;
    s/$lot_mark/<UL>$table_captions<\/UL>/o;
    s/$bbl_mark/$citations/o;
    &add_toc if (/$toc_mark/);
    &add_idx if (/$idx_mark/);
    &replace_cross_references if /$cross_ref_mark/;
    &replace_external_references if /$external_ref_mark/;
    &replace_cite_references;  #just take the if out (not really neccessary)
}  


1;				# Not really necessary...



# T.S.: added the starstar option 
# If a normalized command name exists, return it.
sub normalize {
    local($cmd) = @_;
    local($ncmd);
    if ($ncmd = $normalize{$cmd}) {
	$ncmd}
    elsif ($cmd =~ s/[*]/star/g) {
	$cmd }
    elsif ($cmd =~ s/^@/tex/) {
	$cmd}
    else {$cmd}
}


# does not really belong to harvard.perl, but as a general improvement


sub do_cmd_and {
    join('',"<p>", @_[0]);         #make a paragraph brake of it 
}

sub do_cmd_newline {
    join('',"<br>", @_[0]);
}

