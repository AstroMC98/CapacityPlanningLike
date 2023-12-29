import numpy as np
import plotly.graph_objects as go

class HistoricalCapPlan:
    
    def __init__(self, data, config):
        
        # Configuration File
        self.config = config
        
        # Load in Information from Data
        self.City = data['City']
        
        # Main Properties
        self.ActiveCount = float(data['ActiveCount'])
        self.TargetCount = data['Required FTE']
        self.AnnualLeaveAllocation = data['annual_leave_value']
        self.TOILeaveAllocation = data['toil_leave_value']
        
        self.UnassignedHours = 0
        self.ActivityLog = data[config['activityClass']['billable'] + config['activityClass']['nonbillable']]
        
        self.billable_activity_breakdown()
        self.nonbillable_activity_breakdown()
        self.aggregated_activity_breakdown()
        self.KPI_calculation() 
        
    def billable_activity_breakdown(self):
        self.available = self.ActivityLog.get("available Hours", 0)
        self.onboarding = self.ActivityLog.get("onboarding Hours", 0)
        self.coaching = self.ActivityLog.get("coaching Hours", 0)
        self.meeting = self.ActivityLog.get("team_meeting Hours", 0)
        self.wellness = self.ActivityLog.get("wellness_support Hours", 0)
        self.fbtraining = self.ActivityLog.get("fb_training Hours", 0)
        self.nonfbtraining = self.ActivityLog.get("non-fb-training Hours", 0)
        
    def nonbillable_activity_breakdown(self):
        self.meal = self.ActivityLog.get("meal Hours",0)
        self.breaks = self.ActivityLog.get("break Hours", 0)
        self.leaves = (self.AnnualLeaveAllocation + self.TOILeaveAllocation) * self.config['CityWorkHours'][self.City]['internal']
        
    def aggregated_activity_breakdown(self):
        self.Productive = sum([getattr(self, x, 0) for x in self.config['simulator']['activityTypes']['productive']])
        self.Billable = sum([getattr(self, x, 0) for x in self.config['simulator']['activityTypes']['billable']])
        self.Nonbillable = sum([getattr(self, x, 0) for x in self.config['simulator']['activityTypes']['nonbillable']])
        
        
    def KPI_calculation(self):
        self.TotalHours = self.Billable + self.Nonbillable
        self.RequiredBillableHours = (self.TargetCount * (1-self.config['simulator']['OOO_Target'])) * (self.config['CityWorkHours'][self.City]['client'] * 5) * (.851)
        
        self.WIO = (self.Billable - self.Productive)/(self.Billable)
        self.OOO = self.Nonbillable/self.TotalHours
        self.UtilizationInternal = self.Productive/self.Billable # Based on Internal FTE Count
        self.UtilizationClient = self.Productive/self.RequiredBillableHours# Based on Client FTE Target 
        self.BillableTarget = self.Billable/self.RequiredBillableHours # How many percent did we exceed/lack
        
    def update_activity(self, activity_type, delta):
        self.ActivityLog[activity_type] += delta
        if delta < 0 : # If there has been a reduction in hours allocation
            if activity_type != "available Hours": # If not productive, add to productive
                self.ActivityLog['available Hours'] += -1*delta
            else: # If productive, add to extra hours
                self.ExtraHours += -1*delta
        elif delta > 0: # If there has been additional hours allocation
            if activity_type != 'available Hours': # If not productive, get from Extra Hours
                if self.ExtraHours >= delta: # If there are excess extra hours
                    self.ExtraHours -= delta
                else:
                    
                    if self.ExtraHours > 0: # Exhaust Extra Hours
                        delta -= self.ExtraHours 
                        self.ExtraHours = 0
                    self.ActivityLog['available Hours'] -= delta # Get missing hours from available hours
            else:
                self.ExtraHours -= delta
                
        self.billable_activity_breakdown()
        self.nonbillable_activity_breakdown()
        self.aggregated_activity_breakdown()
        self.KPI_calculation()
        
    def update_FTETarget(self, delta):
        self.TargetCount += delta
        self.ActivityLog['Required FTE'] = self.TargetCount
        
        self.billable_activity_breakdown()
        self.nonbillable_activity_breakdown()
        self.aggregated_activity_breakdown()
        self.KPI_calculation()
        
    def update_ActiveCount(self, delta):
        self.ActiveCount += delta
        self.ActivityLog['ActiveCount'] = self.ActiveCount
        
        # Assume that Theyll Allocate All Hours to Production Hours
        self.ActivityLog['available Hours'] += delta * self.config['CityWorkHours'][self.City]['internal'] * 5
        
        self.billable_activity_breakdown()
        self.nonbillable_activity_breakdown()
        self.aggregated_activity_breakdown()
        self.KPI_calculation()

    def update_AnnualLeaveAllocation(self, delta):
        self.AnnualLeaveAllocation += delta
        self.ActivityLog['annual_leave_value'] = self.AnnualLeaveAllocation
        
        self.billable_activity_breakdown()
        self.nonbillable_activity_breakdown()
        self.aggregated_activity_breakdown()
        self.KPI_calculation()
        
    def update_TOILeaveAllocation(self, value):
        self.TOILeaveAllocation += value
        self.ActivityLog['toil_leave_value'] = self.TOILeaveAllocation
        
        self.billable_activity_breakdown()
        self.nonbillable_activity_breakdown()
        self.aggregated_activity_breakdown()
        self.KPI_calculation()
        
    def generate_waterfall(self):
        import plotly.graph_objects as go
        fig = go.Figure(
            go.Waterfall(
                name = "20",
                orientation = "v",
                measure = [
                    "relative",
                    "relative","relative","relative","relative","relative",
                    "relative",
                    "relative", "relative",
                    "total",
                    "relative",
                    "total",
                    "relative",
                    "total"
                ],
                x = [
                    "Productive",
                    "Onboarding", "Coaching", "Meeting", "Meta Training", "Wellness Support",
                    "Non-Meta Training",
                    "Leave Hours",
                    "Break Hours",
                    "Logged Hours",
                    "OOO Shrinkage",
                    "Billable Hours",
                    "WIO Shrinkage",
                    "Utilization Hours"
                ],
                y = [
                    self.Productive,
                    self.onboarding, self.coaching, self.meeting, self.fbtraining, self.wellness,
                    self.nonfbtraining,
                    self.leaves,
                    self.meal + self.breaks, 
                    0,
                    -1*self.Nonbillable,
                    0,
                    -1*(self.Billable-self.Productive),
                    0
                ],
                connector = {"line":{"color":"rgb(63, 63, 63)"}},
                textposition = "outside",
                showlegend = False,
                text = [
                    f"+{100*self.Productive/self.TotalHours:.2f}%",
                    
                    f"+{100*self.onboarding/self.TotalHours:.2f}%",
                    f"+{100*self.coaching/self.TotalHours:.2f}%",
                    f"+{100*self.meeting/self.TotalHours:.2f}%",
                    f"+{100*self.fbtraining/self.TotalHours:.2f}%",
                    f"+{100*self.wellness/self.TotalHours:.2f}%",
                    f"+{100*self.nonfbtraining/self.TotalHours:.2f}%",
                    
                    f"+{100*self.leaves/self.TotalHours:.2f}%", 
                    f"+{100*(self.meal + self.breaks)/self.TotalHours:.2f}%",  
                    "",
                    f"{100*(self.meal + self.breaks + self.leaves)/self.TotalHours:.2f}%",
                    "",
                    f"{100*(self.Billable-self.Productive)/self.TotalHours:.2f}%",
                    ""
                ],
                customdata = np.stack(
                    (
                        [self.ActiveCount] * 14, # Current Headcount
                        [self.TargetCount] * 14, # Target FTE
                        [self.AnnualLeaveAllocation + self.TOILeaveAllocation] * 14, # Leave Allocation
                        [
                            100 * self.Productive/self.TotalHours,
                            
                            100 * self.onboarding/self.TotalHours, 
                            100 * self.coaching/self.TotalHours, 
                            100 * self.meeting/self.TotalHours, 
                            100 * self.fbtraining/self.TotalHours, 
                            100 * self.wellness/self.TotalHours,
                            100 * self.nonfbtraining/self.TotalHours,
                            
                            100 * self.leaves/self.TotalHours,
                            100 * (self.meal + self.breaks)/self.TotalHours,
                            
                            100 * (self.TotalHours)/self.TotalHours,
                            100 * (self.Nonbillable)/self.TotalHours,
                            100 * (self.TotalHours - self.Nonbillable)/self.TotalHours,
                            100 * ((self.Billable - self.Productive)/self.TotalHours),
                            100 * self.Productive/self.TotalHours
                        ],
                        [
                            self.Productive,
                            
                            self.onboarding, 
                            self.coaching, 
                            self.meeting, 
                            self.fbtraining,
                            self.wellness,
                            self.nonfbtraining,
                            
                            self.leaves,
                            (self.meal + self.breaks),
                            
                            (self.TotalHours),
                            self.Nonbillable,
                            (self.TotalHours - self.Nonbillable),
                            (self.Billable - self.Productive),
                            self.Productive
                        ],
                        [
                            100 * self.Productive/self.Billable,
                            
                            100 * self.onboarding/self.Billable, 
                            100 * self.coaching/self.Billable, 
                            100 * self.meeting/self.Billable, 
                            100 * self.fbtraining/self.Billable, 
                            100 * self.wellness/self.Billable,
                            100 * self.nonfbtraining/self.Billable,
                            
                            0,
                            0,
                            
                            0,
                            0,
                            0,
                            100 * -1 * (self.Productive - self.Billable)/self.Billable,
                            100 * self.Productive/self.Billable
                        ],
                        [
                            "",
                            
                            "", 
                            "", 
                            "", 
                            "",
                            "",
                            "",
                            
                            f"(Annual Leaves + TOI Leaves) x {self.config['CityWorkHours'][self.City]['internal']}<br>",
                            "",

                            "Total Logged Hours<br>",
                            "Leaves + Breaks + Meals<br>",
                            "Total Logged Hours - (Leaves + Breaks + Meals)<br>",
                            "Total Billable Hours - Productive Hours<br>",
                            ""
                        ],
                        
                    ),
                    axis = 1
                ),
                hovertemplate=
                "<b>%{x}</b><br>" +
                "%{customdata[6]}"+
                
                "<br><b>Activity Info</b><br>" +
                "Activity Hours: %{customdata[4]:.2f}<br>" +
                "% of Total Hours: %{customdata[3]:.2f}%<br>" +
                
                "<br><b>HeadCount Info</b><br>" +
                "Actual FTE: %{customdata[0]:.0f}<br>" +
                "Target FTE: %{customdata[1]:.1f}<br>"+
                "Leaves Allocation: %{customdata[2]:.2f}<br>"+
                
                "<br><b>% Allocation wrt Logged Billable Hours:</b><br>" +
                "%{x} : %{customdata[5]:.2f}%"+
                "<extra></extra>",
            )
        )
        
        fig.add_shape(type='line',
                        x0=8.5,
                        y0=self.TotalHours,
                        x1=11.5,
                        y1=self.TotalHours,
                        line=dict(color='Red',),
                        xref='x',
                        yref='y'
        )
        
        fig.add_shape(type='line',
                        x0=5.5,
                        y0=self.TotalHours,
                        x1=13.5,
                        y1=self.TotalHours,
                        line=dict(color='Red',),
                        xref='x',
                        yref='y'
        )
        
        fig.add_shape(type='line',
                        x0=0.5,
                        y0=self.Billable,
                        x1=15.5,
                        y1=self.Billable,
                        line=dict(color='Red',),
                        xref='x',
                        yref='y'
        )
        
        fig.update_layout(
            autosize=True,
            width=1600,
            height=800,)
        
        return fig