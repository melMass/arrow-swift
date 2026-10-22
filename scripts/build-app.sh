#!/bin/sh
# Builds build/Quiver.app. MLX's Metal shaders only compile through xcodebuild, not
# `swift build`, so the app is built by xcodebuild and the bundle assembled around it:
# the executable, its resource bundles (MLX's default.metallib among them) and an
# Info.plist with the document types from apps/Quiver/macos-plist-extras.xml.
#
#   scripts/build-app.sh            ad-hoc signed, runs on this Mac
#   CODESIGN_IDENTITY="Developer ID Application: …" scripts/build-app.sh
#   scripts/build-app.sh --open     and launch it
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
app="$root/apps/Quiver"
out="$root/build"
derived="$out/DerivedData"
bundle="$out/Quiver.app"
version=$(sed -n 's/^version: *"\(.*\)"/\1/p' "$app/app.yaml")

mkdir -p "$out"
(cd "$app" && xcodebuild -scheme Quiver -destination 'platform=macOS,arch=arm64' -configuration Release \
    -derivedDataPath "$derived" -skipPackagePluginValidation -skipMacroValidation build) \
    > "$out/xcodebuild.log" 2>&1 || { tail -30 "$out/xcodebuild.log"; echo "xcodebuild failed; full log: $out/xcodebuild.log"; exit 1; }
products="$derived/Build/Products/Release"

rm -rf "$bundle"
mkdir -p "$bundle/Contents/MacOS" "$bundle/Contents/Resources" "$bundle/Contents/Frameworks"
cp "$products/Quiver" "$bundle/Contents/MacOS/Quiver"
for resources in "$products"/*.bundle; do
    case "$resources" in *Tests.bundle) continue ;; esac
    cp -R "$resources" "$bundle/Contents/Resources/"
done
for framework in "$products"/*.framework "$products"/PackageFrameworks/*.framework; do
    [ -e "$framework" ] && cp -R "$framework" "$bundle/Contents/Frameworks/"
done
install_name_tool -add_rpath @executable_path/../Frameworks "$bundle/Contents/MacOS/Quiver" 2>/dev/null || true

{
    cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Quiver</string>
    <key>CFBundleExecutable</key><string>Quiver</string>
    <key>CFBundleIdentifier</key><string>${BUNDLE_ID:-com.mtb.quiver}</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$version</string>
    <key>CFBundleVersion</key><string>$version</string>
    <key>LSMinimumSystemVersion</key><string>15.0</string>
    <key>LSApplicationCategoryType</key><string>public.app-category.graphics-design</string>
    <key>NSHighResolutionCapable</key><true/>
EOF
    cat "$app/macos-plist-extras.xml"
    printf '</dict>\n</plist>\n'
} > "$bundle/Contents/Info.plist"
plutil -lint "$bundle/Contents/Info.plist" > /dev/null

codesign --force --deep --sign "${CODESIGN_IDENTITY:--}" "$bundle"
echo "$bundle"
[ "$1" = "--open" ] && open "$bundle"
exit 0
