$files = Get-ChildItem "./SuSi/android-platforms"
Set-Location ./SuSi

for ($i=0; $i -lt $files.Count; $i++) {
    $file_name = $files[$i].FullName
    #Write-Output $file_name
    java -cp "lib/weka.jar;soot.jar;soot-infoflow.jar;soot-infoflow-android.jar;bin" de.ecspride.sourcesinkfinder.SourceSinkFinder $file_name permissionMethodWithLabel.pscout out.pscout
}

Set-Location ..\