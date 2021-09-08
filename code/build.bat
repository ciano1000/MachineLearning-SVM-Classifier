@echo off

set application_name=assignment2
rem these will need to be set later
set build_options= -DCEOMHAR_SLOW=1 -DCEOMHAR_INTERNAL=1  -DCEOMHAR_WIN32=1 
set compile_flags=-nologo  -FC -Gm- -MD -GR- -EHa- -O2 -Oi -Zi -WX -W4 -wd4201 -wd4100 -wd4189 -wd4996 
set common_link_flags= opengl32.lib ..\\lib\\glfw3.lib  -opt:ref -incremental:no  
set platform_link_flags= user32.lib Comdlg32.lib gdi32.lib Shell32.lib  -subsystem:windows -ENTRY:mainCRTStartup %common_link_flags%

if not defined DevEnvDir call .\\code\\vcvars.bat
if not exist .\build mkdir .\build 
pushd .\build
cl %build_options% ..\\code\\win32_%application_name%.cpp  %compile_flags% /link %platform_link_flags%  /out:%application_name%.exe
popd