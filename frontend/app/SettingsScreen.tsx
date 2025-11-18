import React, { useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Switch,
} from "react-native";
import { colors } from "../theme/colors";
import SectionHeader from "./components/SectionHeader";

export default function SettingsScreen() {
  const [highRiskNotifications, setHighRiskNotifications] = useState(true);
  const [allEmailAlerts, setAllEmailAlerts] = useState(true);
  const [quietHours, setQuietHours] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(true);

  return (
    <ScrollView style={styles.container}>
      <SectionHeader
        title="Settings"
        subtitle="Manage app preferences and alerts"
      />

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Notifications</Text>

        <View style={styles.settingItem}>
          <View style={styles.settingContent}>
            <Text style={styles.settingLabel}>High-Risk Alerts</Text>
            <Text style={styles.settingDescription}>
              Get notified for high-risk fraud attempts
            </Text>
          </View>
          <Switch
            value={highRiskNotifications}
            onValueChange={setHighRiskNotifications}
            trackColor={{ false: "#ccc", true: colors.green }}
            thumbColor="#fff"
          />
        </View>

        <View style={styles.settingItem}>
          <View style={styles.settingContent}>
            <Text style={styles.settingLabel}>All Email Alerts</Text>
            <Text style={styles.settingDescription}>
              Get notified for all suspicious emails
            </Text>
          </View>
          <Switch
            value={allEmailAlerts}
            onValueChange={setAllEmailAlerts}
            trackColor={{ false: "#ccc", true: colors.green }}
            thumbColor="#fff"
          />
        </View>

        <View style={styles.settingItem}>
          <View style={styles.settingContent}>
            <Text style={styles.settingLabel}>Alert Sound</Text>
            <Text style={styles.settingDescription}>
              Play sound for incoming alerts
            </Text>
          </View>
          <Switch
            value={soundEnabled}
            onValueChange={setSoundEnabled}
            trackColor={{ false: "#ccc", true: colors.green }}
            thumbColor="#fff"
          />
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Quiet Hours</Text>

        <View style={styles.settingItem}>
          <View style={styles.settingContent}>
            <Text style={styles.settingLabel}>Enable Quiet Hours</Text>
            <Text style={styles.settingDescription}>
              Mute notifications during specified times
            </Text>
          </View>
          <Switch
            value={quietHours}
            onValueChange={setQuietHours}
            trackColor={{ false: "#ccc", true: colors.green }}
            thumbColor="#fff"
          />
        </View>

        {quietHours && (
          <View style={styles.quietHoursBox}>
            <View style={styles.timeField}>
              <Text style={styles.timeLabel}>Start Time</Text>
              <Text style={styles.timeValue}>9:00 PM</Text>
            </View>
            <View style={styles.timeField}>
              <Text style={styles.timeLabel}>End Time</Text>
              <Text style={styles.timeValue}>7:00 AM</Text>
            </View>
          </View>
        )}
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Audio</Text>

        <View style={styles.dropdownBox}>
          <Text style={styles.dropdownLabel}>Alert Sound</Text>
          <TouchableOpacity style={styles.dropdown}>
            <Text style={styles.dropdownValue}>Bell Chime</Text>
            <Text style={styles.dropdownArrow}>▼</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.dropdownBox}>
          <Text style={styles.dropdownLabel}>Alert Volume</Text>
          <TouchableOpacity style={styles.dropdown}>
            <Text style={styles.dropdownValue}>High</Text>
            <Text style={styles.dropdownArrow}>▼</Text>
          </TouchableOpacity>
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>About</Text>
        <View style={styles.aboutBox}>
          <View style={styles.aboutItem}>
            <Text style={styles.aboutLabel}>App Version</Text>
            <Text style={styles.aboutValue}>1.0.0</Text>
          </View>
          <View style={styles.aboutItem}>
            <Text style={styles.aboutLabel}>SDK Version</Text>
            <Text style={styles.aboutValue}>54.0.0</Text>
          </View>
        </View>
      </View>

      <TouchableOpacity style={styles.saveButton}>
        <Text style={styles.saveButtonText}>Save Changes</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.white,
    paddingTop: 60,
    paddingHorizontal: 20,
    paddingBottom: 20,
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: "bold",
    color: colors.textDark,
    marginBottom: 12,
  },
  settingItem: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    backgroundColor: colors.alertCard,
    padding: 14,
    borderRadius: 10,
    marginBottom: 10,
  },
  settingContent: {
    flex: 1,
  },
  settingLabel: {
    fontSize: 14,
    fontWeight: "600",
    color: colors.textDark,
    marginBottom: 4,
  },
  settingDescription: {
    fontSize: 12,
    color: "#999",
  },
  quietHoursBox: {
    backgroundColor: colors.alertCard,
    borderRadius: 10,
    padding: 14,
    marginTop: 12,
    flexDirection: "row",
    gap: 12,
  },
  timeField: {
    flex: 1,
  },
  timeLabel: {
    fontSize: 12,
    color: "#999",
    marginBottom: 4,
  },
  timeValue: {
    fontSize: 14,
    fontWeight: "600",
    color: colors.textDark,
  },
  dropdownBox: {
    marginBottom: 14,
  },
  dropdownLabel: {
    fontSize: 12,
    color: "#999",
    marginBottom: 8,
  },
  dropdown: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    backgroundColor: colors.alertCard,
    padding: 14,
    borderRadius: 10,
  },
  dropdownValue: {
    fontSize: 14,
    color: colors.textDark,
    fontWeight: "500",
  },
  dropdownArrow: {
    fontSize: 12,
    color: "#999",
  },
  aboutBox: {
    backgroundColor: colors.alertCard,
    borderRadius: 10,
    padding: 14,
  },
  aboutItem: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingVertical: 8,
  },
  aboutLabel: {
    fontSize: 14,
    color: "#666",
  },
  aboutValue: {
    fontSize: 14,
    fontWeight: "600",
    color: colors.textDark,
  },
  saveButton: {
    backgroundColor: colors.green,
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: "center",
    marginTop: 20,
    marginBottom: 40,
  },
  saveButtonText: {
    fontSize: 16,
    fontWeight: "bold",
    color: colors.textDark,
  },
});
